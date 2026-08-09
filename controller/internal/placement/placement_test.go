package placement

import (
	"strings"
	"testing"
	"time"
)

func gib(n int64) int64 { return n * GiB }

// l4Node is the dev box: g2-standard-24 with 2x L4 24Gi and 94Gi allocatable.
func l4Node(name string, maxResidents int, roles ...string) *Node {
	allowed := map[string]bool{}
	for _, role := range roles {
		allowed[role] = true
	}
	return &Node{
		Name:              name,
		DeviceCount:       2,
		DeviceMemoryBytes: gib(24),
		HostMemoryBytes:   gib(94),
		Roles:             allowed,
		MaxResidents:      maxResidents,
		Product:           "NVIDIA L4",
	}
}

// bigNode is a pool of any shape, for the cases where a workload has to choose.
func bigNode(name string, devices int, deviceGiB int64, maxResidents int, roles ...string) *Node {
	allowed := map[string]bool{}
	for _, role := range roles {
		allowed[role] = true
	}
	return &Node{
		Name:              name,
		DeviceCount:       devices,
		DeviceMemoryBytes: gib(deviceGiB),
		HostMemoryBytes:   gib(340),
		Roles:             allowed,
		MaxResidents:      maxResidents,
	}
}

func trainer(id string, memoryGiB int64) Request {
	return Request{Role: "trainer", WorkerID: id, Memory: gib(memoryGiB)}
}

// booked is a claim with residents already on it, charged cohort by cohort the
// way bookWorker rebuilds one from the workers that reference it. Repeating a
// cohort name puts two residents in it, which is what sharing looks like.
func booked(c *Claim, perDevice int64, cohorts ...string) *Claim {
	for _, cohort := range cohorts {
		c.Book(cohort, perDevice)
	}
	return c
}

func name(s *Selection) string {
	if s == nil {
		return ""
	}
	return s.Claim.Name
}

// There is no sharding: the model is laid out layer by layer over whatever
// devices it gets, so aggregate memory is the only thing that has to add up and
// any device count will do -- including three.
func TestDevicesOnIsPlainCeilingDivision(t *testing.T) {
	node := bigNode("n", 8, 24, 1, "trainer")
	for _, tc := range []struct {
		memoryGiB int64
		want      int
	}{
		{10, 1},
		{24, 1},
		{25, 2},
		{48, 2},
		{60, 3}, // not a power of two, and that is fine
		{192, 8},
		{193, 0}, // more than the pool holds
	} {
		if got := trainer("w", tc.memoryGiB).DevicesOn(node); got != tc.want {
			t.Errorf("DevicesOn(%dGi) = %d, want %d", tc.memoryGiB, got, tc.want)
		}
	}
}

func TestPerDeviceBytesRoundsUp(t *testing.T) {
	req := trainer("w", 30)
	if got, want := req.PerDeviceBytes(2), gib(15); got != want {
		t.Errorf("PerDeviceBytes(2) = %d, want %d", got, want)
	}
	// 30Gi over 4 devices is 7.5Gi each; erring high is the safe direction.
	if got, want := req.PerDeviceBytes(4), (gib(30)+3)/4; got != want {
		t.Errorf("PerDeviceBytes(4) = %d, want %d", got, want)
	}
}

func TestCohortKey(t *testing.T) {
	for _, tc := range []struct {
		name string
		req  Request
		want string
	}{
		{"a named cohort is the cohort", Request{Cohort: "qwen3-0-6b", WorkerID: "job-a"}, "qwen3-0-6b"},
		{"no cohort means a cohort of one", Request{WorkerID: "job-a"}, "job-a"},
	} {
		if got := tc.req.CohortKey(); got != tc.want {
			t.Errorf("%s: CohortKey() = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// The whole of the memory model: memory sums within a cohort, and cohorts take
// turns. Two workers naming one cohort are resident together and their device
// memory adds up; a worker naming a different cohort shares nothing, so it
// waits for a turn rather than being turned away.
func TestCohortsSumWithinAndTakeTurnsBetween(t *testing.T) {
	req := Request{Role: "trainer", Cohort: "qwen3-0-6b", WorkerID: "job-new", Memory: gib(14)}
	fleet := func(perDevice int64, cohorts ...string) *Fleet {
		f := NewFleet()
		f.Nodes["n"] = l4Node("n", 4, "trainer")
		f.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, perDevice, cohorts...)
		return f
	}

	if got := SelectClaim(req, fleet(gib(8), "qwen3-0-6b")); got == nil {
		t.Error("8Gi booked + 14Gi needed fits in 24Gi, but the claim was refused")
	}
	if got := SelectClaim(req, fleet(gib(14), "qwen3-0-6b")); got != nil {
		t.Error("14Gi + 14Gi resident at once exceeds 24Gi of device memory, but the claim was accepted")
	}

	got := SelectClaim(req, fleet(gib(14), "llama-3-8b"))
	if got == nil {
		t.Fatal("refused a claim holding a different cohort; want it admitted as its own cohort")
	}
	if got.Cohort != "qwen3-0-6b" {
		t.Errorf("Cohort = %q, want the request's own", got.Cohort)
	}
}

// Two workers that name no cohort are two cohorts, so they take turns: 14Gi
// twice fits behind 24Gi of device memory where two members of one cohort
// would not.
func TestWorkersWithNoCohortAreEachTheirOwn(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer")
	fleet.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(14), "job-a")

	got := SelectClaim(trainer("job-b", 14), fleet)
	if got == nil {
		t.Fatal("refused a second solo worker; two cohorts take turns rather than summing")
	}
	if got.Cohort != "job-b" {
		t.Errorf("Cohort = %q, want the worker's own id", got.Cohort)
	}
}

// A claim is a bundle of accelerators, not a role or a workload type. On a node
// that accepts both, a sampler joins a trainer's claim and they take turns --
// which is the whole point of running both halves of the loop on one GPU.
func TestSelectClaimMixesRolesOnOneClaim(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer", "sampler")
	fleet.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(6), "lora-trainer")

	sampler := Request{Role: "sampler", WorkerID: "vllm", Memory: gib(8)}
	if got := SelectClaim(sampler, fleet); got == nil {
		t.Fatal("a sampler was refused a trainer's claim on a node that accepts both roles")
	}
}

// Role still gates nodes. A sampler cannot land on a trainer-only pool however
// much room the claim has.
func TestSelectClaimStillHonoursNodeRoles(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer")
	fleet.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(6), "lora-trainer")

	sampler := Request{Role: "sampler", WorkerID: "vllm", Memory: gib(8)}
	if got := SelectClaim(sampler, fleet); got != nil {
		t.Errorf("sampler landed on %s, but the node only accepts trainers", name(got))
	}
}

// Host memory, not the operator's label, is the real bound on how many cohorts
// share a claim: every cohort but the running one is parked in host RAM.
func TestHostMemoryBoundsParkedCohorts(t *testing.T) {
	// Seats for eight and 24Gi of device memory, but only 20Gi of host RAM:
	// 17Gi after headroom, which runs out first.
	node := &Node{
		Name: "n", DeviceCount: 1, DeviceMemoryBytes: gib(24),
		HostMemoryBytes: gib(20), Roles: map[string]bool{"trainer": true}, MaxResidents: 8,
	}
	fleet := NewFleet()
	fleet.Nodes["n"] = node
	fleet.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(10), "job-a")

	// A second solo worker means one cohort runs and the other parks: 10Gi in
	// host RAM, inside the 17Gi budget.
	if got := SelectClaim(trainer("job-b", 10), fleet); got == nil {
		t.Error("refused a second cohort that fits the host budget")
	}

	// A third would leave two cohorts parked, 20Gi, past the budget -- even
	// though the seats and the device memory are both still there.
	fleet.Claims["c"].Book("job-b", gib(10))
	if got := SelectClaim(trainer("job-c", 10), fleet); got != nil {
		t.Error("accepted a third cohort; parking two of them exceeds the host budget")
	}

	// A node that reports no allocatable memory skips the check rather than
	// refusing everything on it.
	node.HostMemoryBytes = 0
	if got := SelectClaim(trainer("job-d", 10), fleet); got == nil {
		t.Error("a node reporting no allocatable memory should skip the host check, not fail it")
	}
}

// The bug the Python controller had: it would only join an *allocated* claim.
// Workers created in the same instant all reconcile before DRA has allocated
// anything, so each cut its own claim and the burst scattered across nodes --
// permanently, because a claim's allocation is immutable.
func TestSelectClaimJoinsAnUnallocatedClaimDuringABurst(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer")
	fleet.Claims["pending"] = booked(&Claim{Name: "pending", DeviceCount: 1}, gib(6), "job-a")

	if got := SelectClaim(trainer("job-b", 6), fleet); name(got) != "pending" {
		t.Errorf("joined %q, want the unallocated claim the burst already cut", name(got))
	}
}

// An unallocated claim is only joinable if some pool this workload fits on
// would produce a claim that shape -- otherwise following it means following it
// nowhere.
func TestSelectClaimIgnoresUnallocatedClaimsOfAnUnreachableShape(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer")
	fleet.Claims["wide"] = booked(&Claim{Name: "wide", DeviceCount: 2}, gib(2), "job-a")

	// A 6Gi workload would cut a 1-device claim, never a 2-device one.
	if got := SelectClaim(trainer("job-b", 6), fleet); got != nil {
		t.Errorf("joined %q, but this workload would never cut a claim that shape", name(got))
	}
}

func TestSelectClaimPrefersAllocatedThenFewestFreeSeats(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 4, "trainer")
	fleet.Claims["pending"] = booked(&Claim{Name: "pending", DeviceCount: 1}, gib(2), "job-a")
	fleet.Claims["roomy"] = booked(&Claim{Name: "roomy", DeviceCount: 1, Node: "n"}, gib(2), "job-b")
	fleet.Claims["tight"] = booked(&Claim{Name: "tight", DeviceCount: 1, Node: "n"}, gib(2), "job-c", "job-d", "job-e")

	// Allocated beats pending, and among allocated the fullest wins so whole
	// accelerators stay free for workers that cannot share.
	if got := SelectClaim(trainer("job-new", 6), fleet); name(got) != "tight" {
		t.Errorf("joined %q, want the fullest allocated claim", name(got))
	}
}

func TestSelectClaimRejectsFullClaims(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", 2, "trainer")
	fleet.Claims["full"] = booked(&Claim{Name: "full", DeviceCount: 1, Node: "n"}, gib(2), "job-a", "job-b")

	if got := SelectClaim(trainer("job-c", 6), fleet); got != nil {
		t.Errorf("joined %q, but max-residents is 2 and both seats are taken", name(got))
	}
}

func TestChoosePoolPrefersTheTightestFit(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = bigNode("l4", 4, 24, 4, "trainer")
	fleet.Nodes["big"] = bigNode("big", 4, 96, 4, "trainer")

	// 20Gi wastes 4Gi on an L4 and 76Gi on the big pool.
	pool := ChoosePool(trainer("w", 20), fleet)
	if pool == nil || pool.Node.Name != "l4" || pool.DeviceCount != 1 {
		t.Fatalf("ChoosePool picked %+v, want 1 device on the L4 pool", pool)
	}

	// 200Gi does not fit four L4s at all, so the big pool is the only answer.
	pool = ChoosePool(trainer("w", 200), fleet)
	if pool == nil || pool.Node.Name != "big" || pool.DeviceCount != 3 {
		t.Fatalf("ChoosePool picked %+v, want 3 devices on the big pool", pool)
	}
}

// The line is fitted to two measured cuda-checkpoint points; this pins it to
// them so a refit that drifts off the measurements is visible.
func TestSwitchCostMatchesTheMeasuredPoints(t *testing.T) {
	for _, tc := range []struct {
		bytes int64
		want  time.Duration
	}{
		{gib(1), 2370 * time.Millisecond},
		{gib(128), 30170 * time.Millisecond},
	} {
		got := SwitchCost(tc.bytes)
		if diff := got - tc.want; diff > 100*time.Millisecond || diff < -100*time.Millisecond {
			t.Errorf("SwitchCost(%dGi) = %s, want about %s", tc.bytes/GiB, got, tc.want)
		}
	}
}

func TestExplain(t *testing.T) {
	empty := NewFleet()
	if got := Explain(trainer("w", 6), empty, ""); !strings.HasPrefix(got, "NoCapacity") {
		t.Errorf("Explain = %q, want NoCapacity for a fleet with no pools", got)
	}

	tooSmall := NewFleet()
	tooSmall.Nodes["n"] = l4Node("n", 4, "trainer") // 2x24Gi
	got := Explain(trainer("w", 200), tooSmall, "")
	if !strings.HasPrefix(got, "NoCapacity") || !strings.Contains(got, "200Gi") {
		t.Errorf("Explain = %q, want NoCapacity naming the 200Gi it could not fit", got)
	}

	full := NewFleet()
	full.Nodes["n"] = l4Node("n", 1, "trainer")
	full.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(20), "job-a")
	got = Explain(trainer("w", 6), full, "pod is unschedulable")
	if !strings.HasPrefix(got, "WaitingForCapacity") || !strings.Contains(got, "pod is unschedulable") {
		t.Errorf("Explain = %q, want WaitingForCapacity carrying the caller's detail", got)
	}
}

func TestReleaseGivesBackTheSeatAndTheMemory(t *testing.T) {
	claim := booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, gib(6), "shared", "shared")
	if got, want := claim.BookedIn("shared"), gib(12); got != want {
		t.Fatalf("BookedIn = %d, want %d", got, want)
	}

	claim.Release("shared", gib(6))
	if got, want := claim.BookedIn("shared"), gib(6); got != want {
		t.Errorf("BookedIn after one release = %d, want %d", got, want)
	}
	if claim.Residents != 1 {
		t.Errorf("Residents = %d, want 1", claim.Residents)
	}

	// The last member out takes the cohort with it, so it stops taking turns.
	claim.Release("shared", gib(6))
	if _, still := claim.Cohorts()["shared"]; still {
		t.Error("an empty cohort is still on the claim; it would keep taking turns")
	}
}
