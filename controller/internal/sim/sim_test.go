package sim

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func run(t *testing.T, yaml string) *Result {
	t.Helper()
	scenario, err := Parse([]byte(yaml))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	res, err := Run(scenario)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	return res
}

func find(t *testing.T, res *Result, id string) WorkerResult {
	t.Helper()
	for _, w := range res.Workers {
		if w.ID == id {
			return w
		}
	}
	t.Fatalf("no workload %q in the result", id)
	return WorkerResult{}
}

// Two adapters naming one cohort hold the GPU together, so neither waits on the
// other and neither is ever switched out for the other. One claim, one cohort,
// zero switches.
func TestOneCohortRunsTogether(t *testing.T) {
	res := run(t, `
quantum: 60s
until: 10m
nodes:
  - {name: l4, devices: 1, deviceMemory: 24Gi, hostMemory: 94Gi, roles: [trainer], maxResidents: 4}
work:
  - {id: a, role: trainer, cohort: qwen, memory: 6Gi, work: 2m}
  - {id: b, role: trainer, cohort: qwen, memory: 6Gi, work: 2m}
`)

	a, b := find(t, res, "a"), find(t, res, "b")
	if a.Claim != b.Claim {
		t.Errorf("adapters landed on %q and %q, want one claim", a.Claim, b.Claim)
	}
	if a.Cohort != b.Cohort {
		t.Errorf("cohorts %q and %q differ, but both named the same base model", a.Cohort, b.Cohort)
	}
	if !a.Done || !b.Done {
		t.Errorf("both should finish: a=%v b=%v", a.Done, b.Done)
	}
	if len(res.Claims) != 1 || res.Claims[0].Cohorts != 1 {
		t.Errorf("claims = %+v, want one claim holding one cohort", res.Claims)
	}
	if res.Claims[0].Switching != 0 {
		t.Errorf("switched for %s, but there is only one cohort to run", res.Claims[0].Switching)
	}
}

// Two cohorts on one claim take turns, so the run costs both of them plus the
// switches in between -- and the accelerators are never held by both at once.
func TestTwoCohortsTakeTurns(t *testing.T) {
	res := run(t, `
quantum: 60s
until: 30m
nodes:
  - {name: l4, devices: 1, deviceMemory: 24Gi, hostMemory: 94Gi, roles: [trainer], maxResidents: 4}
work:
  - {id: fft, role: trainer, memory: 10Gi, work: 3m}
  - {id: lora, role: trainer, cohort: qwen, memory: 6Gi, work: 3m}
`)

	fft, lora := find(t, res, "fft"), find(t, res, "lora")
	if fft.Claim != lora.Claim {
		t.Fatalf("landed on %q and %q, want one shared claim", fft.Claim, lora.Claim)
	}
	if fft.Cohort == lora.Cohort {
		t.Errorf("both in cohort %q, but they share no weights", fft.Cohort)
	}
	if res.Claims[0].Cohorts != 2 {
		t.Errorf("claim holds %d cohorts, want 2", res.Claims[0].Cohorts)
	}
	if fft.Switches == 0 || lora.Switches == 0 {
		t.Errorf("nobody was switched: fft=%d lora=%d", fft.Switches, lora.Switches)
	}
	// Six minutes of work, taken in turns, cannot finish in six minutes.
	if res.Elapsed <= 6*time.Minute {
		t.Errorf("elapsed %s, want longer than the 6m of work once switching is paid for", res.Elapsed)
	}
}

// Role selects nodes, not claims. On a node labelled for both, the sampler
// shares the trainer's accelerator instead of demanding its own.
func TestTrainerAndSamplerShareOneGPU(t *testing.T) {
	res := run(t, `
quantum: 45s
until: 30m
nodes:
  - {name: l4, devices: 1, deviceMemory: 24Gi, hostMemory: 94Gi, roles: [trainer, sampler], maxResidents: 2}
work:
  - {id: trainer, role: trainer, memory: 6Gi, work: 2m}
  - {id: sampler, role: sampler, memory: 8Gi, work: 2m}
`)

	tr, sa := find(t, res, "trainer"), find(t, res, "sampler")
	if tr.Claim != sa.Claim {
		t.Errorf("trainer on %q and sampler on %q, want one claim", tr.Claim, sa.Claim)
	}
	if len(res.Unplaced) > 0 {
		t.Errorf("unplaced: %+v", res.Unplaced)
	}
}

// A workload too big for the pool is refused, and says so rather than sitting
// silently pending.
func TestOversizedWorkloadIsExplained(t *testing.T) {
	res := run(t, `
quantum: 60s
until: 5m
nodes:
  - {name: l4, devices: 1, deviceMemory: 24Gi, hostMemory: 94Gi, roles: [trainer], maxResidents: 4}
work:
  - {id: huge, role: trainer, memory: 200Gi, work: 1m}
`)

	if len(res.Unplaced) != 1 {
		t.Fatalf("unplaced = %+v, want the 200Gi workload", res.Unplaced)
	}
	if !strings.Contains(res.Unplaced[0].Reason, "NoCapacity") {
		t.Errorf("reason = %q, want it to say NoCapacity", res.Unplaced[0].Reason)
	}
}

// A workload that arrives with nowhere to go gets in once a neighbour finishes,
// rather than being refused outright.
func TestAWaitingWorkloadGetsTheSeatBack(t *testing.T) {
	res := run(t, `
quantum: 30s
until: 30m
nodes:
  - {name: l4, devices: 1, deviceMemory: 24Gi, hostMemory: 94Gi, roles: [trainer], maxResidents: 1}
work:
  - {id: first, role: trainer, memory: 20Gi, work: 2m}
  - {id: second, role: trainer, memory: 20Gi, at: 30s, work: 1m}
`)

	second := find(t, res, "second")
	if second.Claim == "" {
		t.Fatalf("second never placed; unplaced = %+v", res.Unplaced)
	}
	if second.Waited == 0 {
		t.Error("second was placed immediately, but the only seat was taken")
	}
	if !second.Done {
		t.Error("second never finished after getting the seat")
	}
}

// The scenarios that ship with the simulator have to actually run, or they are
// documentation that lies.
func TestShippedScenariosRun(t *testing.T) {
	paths, err := filepath.Glob("../../scenarios/*.yaml")
	if err != nil || len(paths) == 0 {
		t.Fatalf("no scenarios found: %v", err)
	}
	for _, path := range paths {
		t.Run(filepath.Base(path), func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			scenario, err := Parse(data)
			if err != nil {
				t.Fatalf("parse: %v", err)
			}
			res, err := Run(scenario)
			if err != nil {
				t.Fatalf("run: %v", err)
			}
			if len(res.Unplaced) > 0 {
				t.Errorf("scenario left work unplaced: %+v", res.Unplaced)
			}
		})
	}
}
