// The spec's end-to-end scenarios live in behavior_test.go, which drives
// these same functions with a simulated DRA. What lives here is only what
// the sim cannot reach: the window where claims exist but DRA has not
// allocated them, the deterministic orderings, and the arithmetic the spec
// pins in its appendix.
package placement

import (
	"strings"
	"testing"
)

func gib(n int64) int64 { return n * GiB }

// l4Node is the dev box: g2-standard-24 with 2x L4 24Gi and 94Gi allocatable.
func l4Node(name string, roles ...string) *Node {
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
		Product:           "NVIDIA L4",
	}
}

// bigNode is a pool of any shape, for the cases where a workload has to choose.
func bigNode(name string, devices int, deviceGiB int64, roles ...string) *Node {
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
	}
}

func trainer(id string, memoryGiB int64) Request {
	// MaxDevices 8: most of these tests exercise the arithmetic of a runtime
	// that declared it can shard; the single-device default is pinned where
	// it matters. The host request mirrors the memory figure: the template
	// asks the node for at least the footprint it may park.
	return Request{Shareable: true, Role: "trainer", WorkerID: id, Memory: gib(memoryGiB), MaxDevices: 8, HostRequestBytes: gib(memoryGiB)}
}

// booked is a claim with workers already assigned to it, charged worker by
// worker the way readFleet rebuilds one from its ledger's seats.
func booked(c *Claim, workers ...string) *Claim {
	for _, worker := range workers {
		c.Book(worker, worker, 0, true)
	}
	return c
}

func name(c *Claim) string {
	if c == nil {
		return ""
	}
	return c.Name
}

// There is no sharding: the model is laid out layer by layer over whatever
// devices it gets, so aggregate memory is the only thing that has to add up and
// any device count will do -- including three.
func TestDevicesOnIsPlainCeilingDivision(t *testing.T) {
	node := bigNode("n", 8, 24, "trainer")
	for _, tc := range []struct {
		memoryGiB int64
		want      int
	}{
		{10, 1},
		{24, 1},
		{25, 2},
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

func TestOwnerKey(t *testing.T) {
	for _, tc := range []struct {
		name string
		req  Request
		want string
	}{
		{"a named owner is the owner", Request{Owner: "qwen3-0-6b", WorkerID: "job-a"}, "qwen3-0-6b"},
		{"no owner means an owner of one", Request{WorkerID: "job-a"}, "job-a"},
	} {
		if got := tc.req.OwnerKey(); got != tc.want {
			t.Errorf("%s: OwnerKey() = %q, want %q", tc.name, got, tc.want)
		}
	}
}

// A pending claim's node is unknown, so nothing about it can be checked
// against anything real: it is never joined. The worker waits for DRA or
// for kube-scheduler's verdict to trigger its own fallback instead.
func TestPendingClaimsAreNotJoined(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = &Node{
		Name: "n", DeviceCount: 1, DeviceMemoryBytes: gib(24), HostMemoryBytes: gib(94),
		Roles: map[string]bool{"trainer": true},
	}
	fleet.Claims["pending"] = booked(&Claim{Name: "pending"}, "job-a")

	if got := SelectClaim(trainer("job-b", 6), fleet); got != nil {
		t.Errorf("joined %q, but an unallocated claim must not be joined", name(got))
	}
}

// Host memory is enforced at the join: a node whose allocatable cannot absorb
// one more parked pod refuses however much device memory remains.
func TestSelectClaimRejectsWhenHostMemoryIsFull(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", "trainer") // 94Gi allocatable
	full := &Claim{Name: "full", DeviceCount: 1, Node: "n"}
	full.Book("job-a", "job-a", gib(45), true)
	full.Book("job-b", "job-b", gib(45), true)
	fleet.Claims["full"] = full

	if got := SelectClaim(trainer("job-c", 6), fleet); got != nil {
		t.Errorf("joined %q, but 90Gi is booked and 6Gi more will not park on 94Gi", name(got))
	}
}

// The catalog is shapes, never free counts: however many claims exist, the
// same tiers come back, because whether anything is free is DRA's question.
func TestCatalogIgnoresOccupancy(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = bigNode("n", 2, 80, "trainer")
	fleet.Claims["c1"] = booked(&Claim{Name: "c1", DeviceCount: 1, Node: "n"}, "job-a")
	fleet.Claims["c2"] = booked(&Claim{Name: "c2", DeviceCount: 1, Node: "n"}, "job-b")

	tiers := Tiers(trainer("w", 10), Catalog(fleet, "trainer"))
	if len(tiers) != 1 || tiers[0].Count != 1 {
		t.Errorf("Tiers = %+v, want the one 80Gi shape regardless of what is booked", tiers)
	}
	if got := Catalog(fleet, "sampler"); len(got) != 0 {
		t.Errorf("Catalog(sampler) = %+v, want empty: the pool is trainer-only", got)
	}
}

// The spec's preference order: fewest assigned workers, then claim name --
// deterministic across reconciles.
func TestSelectClaimPrefersFewestWorkersThenName(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["n"] = l4Node("n", "trainer")
	fleet.Claims["quiet"] = booked(&Claim{Name: "quiet", DeviceCount: 1, Node: "n"}, "job-b")
	fleet.Claims["crowded"] = booked(&Claim{Name: "crowded", DeviceCount: 1, Node: "n"}, "job-c", "job-d", "job-e")

	if got := SelectClaim(trainer("job-new", 6), fleet); name(got) != "quiet" {
		t.Errorf("joined %q, want the claim with the fewest workers", name(got))
	}

	fleet.Claims["a-quiet"] = booked(&Claim{Name: "a-quiet", DeviceCount: 1, Node: "n"}, "job-f")
	if got := SelectClaim(trainer("job-new", 6), fleet); name(got) != "a-quiet" {
		t.Errorf("joined %q, want the tie broken by name", name(got))
	}
}

// Tight fit lives in the tier order: the tier wasting the least device
// memory leads, so small work prefers small devices -- a preference DRA
// honors when it can, never a survey of what is free.
func TestTiersPreferTheTightestFit(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = bigNode("l4", 4, 24, "trainer")
	fleet.Nodes["big"] = bigNode("big", 4, 96, "trainer")

	// 20Gi wastes 4Gi on an L4 and 76Gi on the big pool: both are offered,
	// the L4 shape first.
	tiers := Tiers(trainer("w", 20), Catalog(fleet, "trainer"))
	if len(tiers) != 2 || tiers[0].CeilingBytes != gib(24) || tiers[0].Count != 1 {
		t.Fatalf("Tiers = %+v, want the 24Gi shape leading", tiers)
	}

	// 200Gi does not fit four L4s at all, so the big shape is the only tier.
	tiers = Tiers(trainer("w", 200), Catalog(fleet, "trainer"))
	if len(tiers) != 1 || tiers[0].CeilingBytes != gib(96) || tiers[0].Count != 3 {
		t.Fatalf("Tiers = %+v, want only 3x96Gi", tiers)
	}
}

// Fairness counts distinct owners, not workers.
func TestOwnersCountFairnessUnits(t *testing.T) {
	claim := &Claim{Name: "c", DeviceCount: 1, Node: "n"}
	claim.Book("trainer-a", "job-a", gib(40), true)
	claim.Book("sampler-a", "job-a", gib(20), true)

	if got := claim.Owners(); got != 1 {
		t.Errorf("Owners = %d, want 1: a trainer and sampler of one job are one fairness unit", got)
	}
	if got := claim.Workers(); got != 2 {
		t.Errorf("Workers = %d, want 2", got)
	}
}

// The single-device default: a worker that declares no shape is never given
// a claim wider than one device, and never joins a wide claim.
func TestDefaultShapeIsSingleDevice(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = bigNode("l4", 4, 24, "trainer")

	// 60Gi would take three L4s, but an undeclared runtime drives one device.
	undeclared := Request{Role: "trainer", WorkerID: "w", Memory: gib(60)}
	if tiers := Tiers(undeclared, Catalog(fleet, "trainer")); len(tiers) != 0 {
		t.Fatalf("Tiers = %+v for a single-device runtime that fits no single device", tiers)
	}

	// Declaring the shape is what unlocks the wider claim.
	declared := undeclared
	declared.MaxDevices = 4
	if tiers := Tiers(declared, Catalog(fleet, "trainer")); len(tiers) != 1 || tiers[0].Count != 3 {
		t.Fatalf("Tiers = %+v, want 3 devices once the runtime declares them", tiers)
	}
}

// A pending worker's reason distinguishes "busy, retry" from "too small,
// never" -- the spec's actionable-reason requirement.
func TestExplain(t *testing.T) {
	empty := NewFleet()
	if got := Explain(trainer("w", 6), empty, ""); !strings.HasPrefix(got, "NoCapacity") {
		t.Errorf("Explain = %q, want NoCapacity for a fleet with no pools", got)
	}

	tooSmall := NewFleet()
	tooSmall.Nodes["n"] = l4Node("n", "trainer") // 2x24Gi
	got := Explain(trainer("w", 200), tooSmall, "")
	if !strings.HasPrefix(got, "NoCapacity") || !strings.Contains(got, "200Gi") {
		t.Errorf("Explain = %q, want NoCapacity naming the 200Gi it could not fit", got)
	}

	full := NewFleet()
	full.Nodes["n"] = l4Node("n", "trainer")
	full.Claims["c"] = booked(&Claim{Name: "c", DeviceCount: 1, Node: "n"}, "job-a")
	got = Explain(trainer("w", 6), full, "pod is unschedulable")
	if !strings.HasPrefix(got, "WaitingForCapacity") || !strings.Contains(got, "pod is unschedulable") {
		t.Errorf("Explain = %q, want WaitingForCapacity carrying the caller's detail", got)
	}
}

// The tier order must survive the trip across nodes: the smallest adequate
// shape's pools take the top weight, middles descend, and the largest shape
// gets no rung -- it is the baseline nothing needs to outrank.
func TestPreferTightFitRanksSmallShapesFirst(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4-a"] = l4Node("l4-a", "trainer")
	fleet.Nodes["l4-b"] = l4Node("l4-b", "trainer")
	fleet.Nodes["mid"] = bigNode("mid", 1, 48, "trainer")
	fleet.Nodes["h100"] = bigNode("h100", 1, 80, "trainer")

	prefs := PreferTightFit(fleet, Request{Role: "trainer", WorkerID: "w", Memory: gib(10)})

	if len(prefs) != 2 {
		t.Fatalf("prefs = %+v, want two rungs: L4s, then the 48Gi pool", prefs)
	}
	if prefs[0].Weight != 100 || strings.Join(prefs[0].Nodes, ",") != "l4-a,l4-b" {
		t.Errorf("top rung = %+v, want the L4 pools at weight 100", prefs[0])
	}
	if prefs[1].Weight != 50 || strings.Join(prefs[1].Nodes, ",") != "mid" {
		t.Errorf("middle rung = %+v, want the 48Gi pool at weight 50", prefs[1])
	}
}

// One adequate shape means no order to express: no rungs for a request only
// the big pools fit, and none on a single-shape fleet.
func TestPreferTightFitIsSilentWithoutAChoice(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = l4Node("l4", "trainer")
	fleet.Nodes["h100"] = bigNode("h100", 1, 80, "trainer")

	if prefs := PreferTightFit(fleet, Request{Role: "trainer", WorkerID: "w", Memory: gib(60)}); prefs != nil {
		t.Fatalf("prefs = %+v, want none: only the 80Gi shape fits 60Gi", prefs)
	}

	uniform := NewFleet()
	uniform.Nodes["a"] = l4Node("a", "trainer")
	uniform.Nodes["b"] = l4Node("b", "trainer")
	if prefs := PreferTightFit(uniform, Request{Role: "trainer", WorkerID: "w", Memory: gib(10)}); prefs != nil {
		t.Fatalf("prefs = %+v, want none on a single-shape fleet", prefs)
	}
}

// A device the driver reports at 79.65Gi and one it reports at 80Gi are the
// same shape: one catalog entry, sized by the smaller so the fit holds on
// either, and both nodes earn the same tight-fit rung. Bucketed by whole
// GiB, so the tier list is the same however the map iterates.
func TestCatalogBucketsSubGiBSizes(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["exact"] = bigNode("exact", 1, 80, "trainer")
	fleet.Nodes["short"] = bigNode("short", 1, 80, "trainer")
	fleet.Nodes["short"].DeviceMemoryBytes = gib(80) - 360*1024*1024 // ~79.65Gi
	fleet.Nodes["big"] = bigNode("big", 1, 96, "trainer")

	catalog := Catalog(fleet, "trainer")
	if len(catalog) != 2 {
		t.Fatalf("Catalog = %+v, want two shapes: one 80Gi bucket and 96Gi", catalog)
	}
	for _, size := range catalog {
		if CeilGiB(size.DeviceMemoryBytes) == 80 && size.DeviceMemoryBytes != fleet.Nodes["short"].DeviceMemoryBytes {
			t.Errorf("80Gi bucket sized %d, want the smaller device %d", size.DeviceMemoryBytes, fleet.Nodes["short"].DeviceMemoryBytes)
		}
	}

	// 10Gi: the 80Gi tier leads, and its rung names both 80Gi-class nodes.
	prefs := PreferTightFit(fleet, trainer("w", 10))
	if len(prefs) != 1 || strings.Join(prefs[0].Nodes, ",") != "exact,short" {
		t.Fatalf("PreferTightFit = %+v, want one rung naming exact and short", prefs)
	}
}

// Binpack is the default: an unset strategy shares first and cuts a claim
// only when no ledger can seat the worker.
func TestDefaultStrategyIsBinPack(t *testing.T) {
	got, err := ParseStrategy("")
	if err != nil || got != StrategyBinPack {
		t.Fatalf("ParseStrategy(\"\") = %q, %v; want binpack", got, err)
	}
	if _, err := ParseStrategy("sideways"); err == nil {
		t.Error("an unknown strategy was accepted")
	}
}

// A MultiGPU request is an exact group: the count is the request's, not the
// memory's, each device must hold the per-device peak, and only sizes some
// node offers that many of can serve it. Memory division never applies.
func TestMultiGPURequestsCompileToExactCountTiers(t *testing.T) {
	fleet := NewFleet()
	fleet.Nodes["l4"] = l4Node("l4", "trainer")
	fleet.Nodes["h100"] = bigNode("h100", 8, 80, "trainer")

	four := Request{Role: "trainer", WorkerID: "tp4", Memory: gib(40), Devices: 4, MaxDevices: 4}
	tiers := Tiers(four, Catalog(fleet, "trainer"))
	if len(tiers) != 1 || tiers[0].Count != 4 || tiers[0].FloorBytes != gib(40) || CeilGiB(tiers[0].CeilingBytes) != 80 {
		t.Fatalf("Tiers = %+v, want one t4x80 tier with a 40Gi floor; no L4 node has four devices", tiers)
	}
	// Under memory division 40Gi would be one 80Gi device. Never for a group.
	if got := four.DevicesOn(fleet.Nodes["h100"]); got != 4 {
		t.Errorf("DevicesOn(h100) = %d, want the requested 4", got)
	}
	if got := four.DevicesOn(fleet.Nodes["l4"]); got != 0 {
		t.Errorf("DevicesOn(l4) = %d, want 0: two devices cannot hold a group of four", got)
	}

	// Two devices of 20Gi fit either size; the tighter fit is preferred.
	two := Request{Role: "trainer", WorkerID: "tp2", Memory: gib(20), Devices: 2, MaxDevices: 2}
	tiers = Tiers(two, Catalog(fleet, "trainer"))
	if len(tiers) != 2 || tiers[0].Name != "t2x24" || tiers[1].Name != "t2x80" {
		t.Fatalf("Tiers = %+v, want t2x24 before t2x80", tiers)
	}

	// Too much per device for any size: never, and the reason says why.
	huge := Request{Role: "trainer", WorkerID: "big", Memory: gib(100), Devices: 2, MaxDevices: 2}
	if got := Explain(huge, fleet, ""); !strings.Contains(got, "NoCapacity: needs 2 device(s) of 100Gi") {
		t.Errorf("Explain = %q, want the group's shape in the reason", got)
	}
}
