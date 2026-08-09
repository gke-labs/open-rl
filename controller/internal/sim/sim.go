// Package sim runs the scheduler against a cluster that does not exist.
//
// It drives internal/placement directly -- the same functions the controller
// calls -- so what it shows is the real decision, not a model of one. What it
// adds is time: workloads arrive, cohorts take turns on the claims they land
// on, switches cost what the measured cost model says they cost, and workloads
// finish and give their seats back.
//
// The point is to answer "what happens if" without a cluster: half the GPUs,
// twice the workers, one big model beside ten small ones.
package sim

import (
	"fmt"
	"sort"
	"time"

	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// Scenario is a cluster and a stream of work to throw at it.
//
// Every duration and every memory figure is a string in the units a human
// would write -- "90s", "24Gi" -- because a scenario is something you hand-edit.
type Scenario struct {
	// Quantum is how long a cohort holds the accelerators before the timeslicer
	// offers them to the next one. Defaults to 60s.
	Quantum string `json:"quantum,omitempty"`
	// Until stops the run even if work remains, so a scenario that cannot drain
	// still reports rather than looping. Defaults to 30m.
	Until string         `json:"until,omitempty"`
	Nodes []NodeSpec     `json:"nodes"`
	Work  []WorkloadSpec `json:"work"`
}

// NodeSpec is one accelerator pool, as the operator would label it.
type NodeSpec struct {
	Name string `json:"name"`
	// Devices and DeviceMemory are what the DRA driver would report.
	Devices      int    `json:"devices"`
	DeviceMemory string `json:"deviceMemory"`
	HostMemory   string `json:"hostMemory"`
	// Roles is openrl.io/trainer and openrl.io/sampler. A node carrying both
	// can host a trainer and a sampler on the same accelerator, by turns.
	Roles []string `json:"roles"`
	// MaxResidents is openrl.io/max-residents.
	MaxResidents int `json:"maxResidents"`
}

// WorkloadSpec is one OpenRLWorker, plus the two things a real spec has no
// reason to carry: when it shows up and how long it needs.
type WorkloadSpec struct {
	ID     string `json:"id"`
	Role   string `json:"role"`
	Cohort string `json:"cohort,omitempty"`
	Memory string `json:"memory"`
	// At is when the request is submitted. Empty means at the start.
	At string `json:"at,omitempty"`
	// Work is accelerator time the workload needs to finish. Empty means it
	// never finishes and holds its seat for the whole run.
	Work string `json:"work,omitempty"`
}

// Result is what a run produced.
type Result struct {
	Events   []Event
	Workers  []WorkerResult
	Claims   []ClaimResult
	Unplaced []Unplaced
	Elapsed  time.Duration
}

// Event is one thing that happened, in order.
type Event struct {
	At   time.Duration
	Kind string // place | cut | turn | done | wait
	Text string
}

// WorkerResult is how one workload fared.
type WorkerResult struct {
	ID       string
	Cohort   string
	Claim    string
	Node     string
	Devices  int
	Waited   time.Duration // from submission to placement
	Ran      time.Duration // accelerator time actually received
	Needed   time.Duration
	Switches int
	Done     bool
}

// ClaimResult is how one claim was used.
type ClaimResult struct {
	Name string
	Node string
	// Devices is the width of the bundle.
	Devices int
	// Cohorts is how many distinct cohorts ever sat on it.
	Cohorts int
	// Busy is time spent running, Switching is time spent parking and
	// restoring. The rest of the run it was idle.
	Busy      time.Duration
	Switching time.Duration
}

// Unplaced is a workload the scheduler refused, and why.
type Unplaced struct {
	ID     string
	Reason string
}

// worker is a workload's live state during a run.
type worker struct {
	spec    WorkloadSpec
	request placement.Request
	// at and work are spec.At and spec.Work, parsed once.
	at, work time.Duration
	cohort   string
	claim    *placement.Claim
	perDev   int64
	placedAt time.Duration
	ran      time.Duration
	switches int
	done     bool
}

// claimState is the turn-taking state for one claim: who holds the
// accelerators and how much of their turn is left.
type claimState struct {
	claim     *placement.Claim
	active    string // the cohort currently holding the accelerators
	order     []string
	seen      map[string]bool
	busy      time.Duration
	switching time.Duration
}

// Run plays the scenario out and reports what happened.
//
// Time advances one quantum at a time. Within a step every claim grants its
// turn to one cohort; the members of that cohort all make progress, and
// everybody else is parked. A switch is charged against the incoming cohort's
// turn, which is what makes a short quantum visibly wasteful.
func Run(s Scenario) (*Result, error) {
	fleet, err := buildFleet(s.Nodes)
	if err != nil {
		return nil, err
	}
	workers, err := buildWorkers(s.Work)
	if err != nil {
		return nil, err
	}

	quantum, err := duration(s.Quantum, 60*time.Second)
	if err != nil {
		return nil, fmt.Errorf("quantum: %w", err)
	}
	until, err := duration(s.Until, 30*time.Minute)
	if err != nil {
		return nil, fmt.Errorf("until: %w", err)
	}

	res := &Result{}
	states := map[string]*claimState{}
	pending := append([]*worker(nil), workers...)

	for now := time.Duration(0); now <= until; now += quantum {
		pending = admit(res, fleet, states, pending, now)
		if allDone(workers) {
			res.Elapsed = now
			break
		}
		advance(res, fleet, states, workers, quantum, now)
		res.Elapsed = now + quantum
	}

	for _, w := range pending {
		reason := placement.Explain(w.request, fleet, "")
		res.Unplaced = append(res.Unplaced, Unplaced{ID: w.spec.ID, Reason: reason})
	}
	res.Workers = summarizeWorkers(workers)
	res.Claims = summarizeClaims(states)
	return res, nil
}

// admit tries to place every workload that has arrived, in submission order.
// Anything it cannot place stays pending and is retried next step, which is how
// a workload waits for a seat that a finishing neighbour is about to free.
func admit(res *Result, fleet *placement.Fleet, states map[string]*claimState, pending []*worker, now time.Duration) []*worker {
	var stillPending []*worker
	for _, w := range pending {
		if w.at > now {
			stillPending = append(stillPending, w)
			continue
		}
		sel, cut := place(fleet, w.request)
		if sel == nil {
			stillPending = append(stillPending, w)
			if now == w.at {
				res.log(now, "wait", "%s has nowhere to go yet: %s", w.spec.ID, placement.Explain(w.request, fleet, ""))
			}
			continue
		}
		sel.Claim.Book(sel.Cohort, sel.PerDeviceBytes)
		w.claim, w.cohort, w.perDev, w.placedAt = sel.Claim, sel.Cohort, sel.PerDeviceBytes, now

		st, ok := states[sel.Claim.Name]
		if !ok {
			st = &claimState{claim: sel.Claim, seen: map[string]bool{}}
			states[sel.Claim.Name] = st
		}
		if !st.seen[sel.Cohort] {
			st.seen[sel.Cohort] = true
			st.order = append(st.order, sel.Cohort)
		}

		verb := "joined"
		if cut {
			verb = "cut"
		}
		res.log(now, "place", "%s %s %s (%dx%s/dev) as cohort %q",
			w.spec.ID, verb, sel.Claim.Name, sel.Claim.DeviceCount, gib(sel.PerDeviceBytes), sel.Cohort)
	}
	return stillPending
}

// place is the controller's assign step: join a claim, or cut a new one.
func place(fleet *placement.Fleet, req placement.Request) (*placement.Selection, bool) {
	if sel := placement.SelectClaim(req, fleet); sel != nil {
		return sel, false
	}
	pool := placement.ChoosePool(req, fleet)
	if pool == nil {
		return nil, false
	}
	claim := &placement.Claim{
		Name:        fmt.Sprintf("claim-%s", req.WorkerID),
		DeviceCount: pool.DeviceCount,
		// The simulated scheduler binds immediately; a real one waits for DRA.
		Node: pool.Node.Name,
	}
	fleet.Claims[claim.Name] = claim
	return &placement.Selection{
		Claim:          claim,
		Cohort:         req.CohortKey(),
		PerDeviceBytes: req.PerDeviceBytes(pool.DeviceCount),
	}, true
}

// advance runs one quantum: every claim gives a turn to one cohort, its members
// make progress, and anything that finishes gives its seat back.
func advance(res *Result, fleet *placement.Fleet, states map[string]*claimState, workers []*worker, quantum, now time.Duration) {
	for _, name := range sortedKeys(states) {
		st := states[name]
		live := liveCohorts(workers, st.claim)
		if len(live) == 0 {
			continue
		}

		next := nextCohort(st, live)
		useful := quantum
		// A switch is only paid when somebody has to be evicted to make room.
		// Loading the first cohort onto an idle claim costs what starting any
		// process costs, which is not a scheduling decision and not what this
		// is measuring: switches here count preemptions.
		if next != st.active && st.active != "" {
			// Restoring the incoming cohort eats into its own turn. Parked bytes
			// are what the outgoing cohort has to move out of the way.
			cost := placement.SwitchCost(st.claim.BookedIn(next) * int64(st.claim.DeviceCount))
			if cost > quantum {
				cost = quantum
			}
			st.switching += cost
			useful -= cost
			res.log(now, "turn", "%s -> cohort %q (switch %s)", st.claim.Name, next, round(cost))
			for _, w := range workers {
				if w.claim == st.claim && !w.done {
					w.switches++
				}
			}
		}
		st.active = next
		st.busy += useful

		for _, w := range workers {
			if w.claim != st.claim || w.cohort != st.active || w.done {
				continue
			}
			w.ran += useful
			if w.work > 0 && w.ran >= w.work {
				w.done = true
				st.claim.Release(w.cohort, w.perDev)
				res.log(now+quantum, "done", "%s finished after %s on the accelerators", w.spec.ID, round(w.ran))
			}
		}
		if !anyLive(workers, st.claim, st.active) {
			st.active = ""
		}
		// An empty claim is deallocated and its accelerators go back to the
		// pool. The claimState stays behind so the report can still account for
		// what happened on it.
		if st.claim.Residents <= 0 {
			delete(fleet.Claims, st.claim.Name)
		}
	}
}

// nextCohort is round-robin over the cohorts that still have live members,
// which is the rule the node-local timeslicer follows.
func nextCohort(st *claimState, live map[string]bool) string {
	var ring []string
	for _, c := range st.order {
		if live[c] {
			ring = append(ring, c)
		}
	}
	if len(ring) == 0 {
		return ""
	}
	for i, c := range ring {
		if c == st.active {
			return ring[(i+1)%len(ring)]
		}
	}
	return ring[0]
}

func liveCohorts(workers []*worker, claim *placement.Claim) map[string]bool {
	live := map[string]bool{}
	for _, w := range workers {
		if w.claim == claim && !w.done {
			live[w.cohort] = true
		}
	}
	return live
}

func anyLive(workers []*worker, claim *placement.Claim, cohort string) bool {
	for _, w := range workers {
		if w.claim == claim && w.cohort == cohort && !w.done {
			return true
		}
	}
	return false
}

func allDone(workers []*worker) bool {
	for _, w := range workers {
		if !w.done {
			return false
		}
	}
	return true
}

func summarizeWorkers(workers []*worker) []WorkerResult {
	out := make([]WorkerResult, 0, len(workers))
	for _, w := range workers {
		r := WorkerResult{
			ID: w.spec.ID, Cohort: w.cohort, Ran: w.ran,
			Needed: w.work, Switches: w.switches, Done: w.done,
		}
		if w.claim != nil {
			r.Claim, r.Node, r.Devices = w.claim.Name, w.claim.Node, w.claim.DeviceCount
			r.Waited = w.placedAt - w.at
		}
		out = append(out, r)
	}
	return out
}

func summarizeClaims(states map[string]*claimState) []ClaimResult {
	out := make([]ClaimResult, 0, len(states))
	for _, name := range sortedKeys(states) {
		st := states[name]
		out = append(out, ClaimResult{
			Name: st.claim.Name, Node: st.claim.Node, Devices: st.claim.DeviceCount,
			Cohorts: len(st.order), Busy: st.busy, Switching: st.switching,
		})
	}
	return out
}

func (r *Result) log(at time.Duration, kind, format string, args ...any) {
	r.Events = append(r.Events, Event{At: at, Kind: kind, Text: fmt.Sprintf(format, args...)})
}

func sortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

func round(d time.Duration) time.Duration { return d.Round(100 * time.Millisecond) }

func gib(bytes int64) string { return fmt.Sprintf("%dGi", placement.CeilGiB(bytes)) }
