// What we expect of placement, end to end, played the way the controller
// plays it -- cut a tiered claim, DRA answers instantly, an unsatisfied
// claim falls back to sharing on kube-scheduler's verdict. The memory
// figures are the estimator's real outputs: the API server's tier table says
// 10Gi for the L4 tier and 60Gi for the 80Gi tier, and the hardware shapes
// are the ones we run (2x L4 24Gi dev box, single 80Gi devices).
//
//   - free GPUs are used before anyone shares
//   - workers share a claim whether or not their sums fit; only independent
//     fit matters
//   - a node whose host memory is spoken for makes the next worker wait,
//     and a departure frees the memory
//   - role labels are policy: they hide hardware from workers
//   - host memory bounds how many suspended workers a node carries
//   - a worker no single node can hold stays waiting, with a reason
//   - placed workers stay put when capacity frees; new capacity serves new
//     work
//
// Who is resident at any moment is the timeslicer's job, tested in
// tests/test_accel_timeslicer.py. The pipeline against a real API server is
// hack/kind-smoke.sh.
package placement

import (
	"sort"
	"strings"
	"testing"
)

// cluster drives placement the way the controller does -- cut a tiered
// claim, and fall back to sharing when it goes unsatisfied -- with the
// simulation liberty that DRA and kube-scheduler answer immediately. The
// free-device survey below is deliberately here and not in the package: it
// is DRA's knowledge, played by the sim.
type cluster struct {
	t      *testing.T
	fleet  *Fleet
	placed map[string]*Claim
}

func newCluster(t *testing.T, nodes ...*Node) *cluster {
	t.Helper()
	fleet := NewFleet()
	for _, n := range nodes {
		fleet.Nodes[n.Name] = n
	}
	return &cluster{t: t, fleet: fleet, placed: map[string]*Claim{}}
}

// arrive places one worker and reports where it landed: a claim name, or ""
// while it waits. The worker cuts a tiered claim; the sim plays DRA (first
// tier with a node holding enough free devices wins, and kube-scheduler's
// host-memory check rides along); an unsatisfied claim falls straight to
// SelectClaim, the verdict played as instant.
func (c *cluster) arrive(req Request) string {
	c.t.Helper()
	for _, tier := range Tiers(req, Catalog(c.fleet, req.Role)) {
		for _, node := range c.nodesByName() {
			if !node.Accepts(req.Role) ||
				node.DeviceMemoryBytes < tier.FloorBytes || node.DeviceMemoryBytes > tier.CeilingBytes {
				continue
			}
			if c.freeDevices(node) < tier.Count {
				continue
			}
			if node.HostMemoryBytes > 0 && c.fleet.NodeHostBytes(node)+req.HostRequestBytes > node.HostMemoryBytes {
				continue
			}
			claim := &Claim{Name: "claim-" + req.WorkerID, DeviceCount: tier.Count, Node: node.Name}
			claim.Book(req.WorkerID, req.OwnerKey(), req.HostRequestBytes, req.Shareable)
			c.fleet.Claims[claim.Name] = claim
			c.placed[req.WorkerID] = claim
			return claim.Name
		}
	}
	if join := SelectClaim(req, c.fleet); join != nil {
		join.Book(req.WorkerID, req.OwnerKey(), req.HostRequestBytes, req.Shareable)
		c.placed[req.WorkerID] = join
		return join.Name
	}
	return ""
}

// freeDevices is the sim's copy of DRA's ledger: how many of a node's
// devices no allocated claim holds.
func (c *cluster) freeDevices(node *Node) int {
	free := node.DeviceCount
	for _, claim := range c.fleet.Claims {
		if claim.Node == node.Name {
			free -= claim.DeviceCount
		}
	}
	return free
}

func (c *cluster) nodesByName() []*Node {
	names := make([]string, 0, len(c.fleet.Nodes))
	for name := range c.fleet.Nodes {
		names = append(names, name)
	}
	sort.Strings(names)
	nodes := make([]*Node, len(names))
	for i, name := range names {
		nodes[i] = c.fleet.Nodes[name]
	}
	return nodes
}

// leave releases a worker's seat; an emptied claim returns its accelerators.
func (c *cluster) leave(workerID string) {
	c.t.Helper()
	claim := c.placed[workerID]
	if claim == nil {
		c.t.Fatalf("%s was never placed", workerID)
	}
	claim.Release(workerID)
	delete(c.placed, workerID)
	if claim.Workers() == 0 {
		delete(c.fleet.Claims, claim.Name)
	}
}

// waitingReason is why an unplaced worker is waiting.
func (c *cluster) waitingReason(req Request) string {
	return Explain(req, c.fleet, "")
}

func l4Box(name string) *Node {
	return &Node{
		Name: name, DeviceCount: 2, DeviceMemoryBytes: gib(24), HostMemoryBytes: gib(94),
		Roles: map[string]bool{"trainer": true, "sampler": true}, Product: "NVIDIA L4",
	}
}

func gpu80(name string, roles ...string) *Node {
	allowed := map[string]bool{}
	for _, role := range roles {
		allowed[role] = true
	}
	return &Node{
		Name: name, DeviceCount: 1, DeviceMemoryBytes: gib(80), HostMemoryBytes: gib(180),
		Roles: allowed,
	}
}

func TestWorkersSpreadAcrossFreeGPUs(t *testing.T) {
	c := newCluster(t, gpu80("node-a", "trainer"), gpu80("node-b", "trainer"))

	// Two 80Gi-tier estimates (the API server's table says 60Gi) while both
	// GPUs are free: a claim each, on different nodes.
	a := c.arrive(trainer("w1", 60))
	b := c.arrive(trainer("w2", 60))
	if a == "" || b == "" || a == b {
		t.Fatalf("placed on %q and %q, want separate claims while devices are free", a, b)
	}
	if c.placed["w1"].Node == c.placed["w2"].Node {
		t.Errorf("both claims landed on %s, want separate nodes", c.placed["w1"].Node)
	}
}

// An FFT trainer and its sampler under contention: whether their sum would
// fit (45+25 on 80Gi) or not (55+35), V1 places them identically, because
// each only has to fit alone -- they take exclusive turns at runtime.
func TestWorkersShareAClaimWhetherOrNotTheirSumFits(t *testing.T) {
	for name, pair := range map[string][2]int64{
		"sum fits":     {45, 25},
		"sum does not": {55, 35},
	} {
		t.Run(name, func(t *testing.T) {
			c := newCluster(t, gpu80("gpu", "trainer", "sampler"))
			first := c.arrive(trainer("trainer", pair[0]))
			second := c.arrive(Request{Shareable: true, Role: "sampler", WorkerID: "sampler", Memory: gib(pair[1])})
			if first == "" || first != second {
				t.Fatalf("placed on %q and %q, want one shared claim", first, second)
			}
		})
	}
}

// Host memory is the seat ceiling: the third worker fits the GPU but its pod
// request will not park on the node, so it waits, and takes the freed memory
// the moment a departure returns it. The claim outlives its first worker.
func TestAFullNodeMakesTheNextWorkerWaitForHostMemory(t *testing.T) {
	c := newCluster(t, gpu80("gpu", "trainer")) // 180Gi allocatable
	c.arrive(trainer("j1", 70))
	shared := c.arrive(trainer("j2", 70))

	if got := c.arrive(trainer("j3", 50)); got != "" {
		t.Fatalf("j3 was seated on %q, but 190Gi of pod requests will not park on 180Gi", got)
	}
	c.leave("j1")
	if got := c.arrive(trainer("j3", 50)); got != shared {
		t.Fatalf("j3 landed on %q, want the memory freed on %q -- the claim outlives its first worker", got, shared)
	}
}

// Labels decide what OpenRL may use: a free GPU on a trainer-only node is
// invisible to a sampler, and the reason names the policy.
func TestRoleLabelsHideFreeHardware(t *testing.T) {
	c := newCluster(t, gpu80("trainer-only", "trainer"))

	sampler := Request{Shareable: true, Role: "sampler", WorkerID: "sampler", Memory: gib(10)}
	if got := c.arrive(sampler); got != "" {
		t.Fatalf("sampler landed on %q, but the operator allowed no sampler nodes", got)
	}
	if reason := c.waitingReason(sampler); !strings.Contains(reason, "no enabled node accepts sampler") {
		t.Errorf("reason = %q, want the policy named, not capacity", reason)
	}
}

// GPU memory fits but host memory does not: suspended workers park in host
// RAM, so a thin-host node refuses the worker that would overflow it -- until
// a departure frees the memory.
func TestHostMemoryBoundsAdmission(t *testing.T) {
	thin := gpu80("thin-host", "trainer")
	thin.HostMemoryBytes = gib(40) // two 20Gi pod requests fit; a third does not
	c := newCluster(t, thin)

	c.arrive(trainer("w1", 20))
	if got := c.arrive(trainer("w2", 20)); got == "" {
		t.Fatal("w2 parks only 20Gi, inside the budget, and should have joined")
	}
	if got := c.arrive(trainer("w3", 20)); got != "" {
		t.Fatalf("w3 was seated on %q, but parking two 20Gi workers exceeds the 34Gi budget", got)
	}
	c.leave("w1")
	if got := c.arrive(trainer("w3", 20)); got == "" {
		t.Fatal("w3 still refused after the departure freed host memory")
	}
}

// Capacity that exists only across nodes places nothing; the same memory on
// one node places as a multi-device claim.
func TestAWorkerNoSingleNodeCanHoldStaysWaiting(t *testing.T) {
	across := newCluster(t, gpu80("node-a", "trainer"), gpu80("node-b", "trainer"))
	big := trainer("big", 140)
	if got := across.arrive(big); got != "" {
		t.Fatalf("140Gi landed on %q, but no single node can hold it", got)
	}
	if reason := across.waitingReason(big); !strings.Contains(reason, "NoCapacity") {
		t.Errorf("reason = %q, want NoCapacity: waiting would never help", reason)
	}

	oneNode := newCluster(t, &Node{
		Name: "wide", DeviceCount: 2, DeviceMemoryBytes: gib(80), HostMemoryBytes: gib(360),
		Roles: map[string]bool{"trainer": true},
	})
	if oneNode.arrive(big) == "" || oneNode.placed["big"].DeviceCount != 2 {
		t.Fatalf("placed = %+v, want a two-device claim on the one node that fits it", oneNode.placed["big"])
	}
}

// V1 does not rebalance: the pair placed under contention keeps sharing after
// a GPU frees, and the freed GPU serves new work instead.
func TestPlacedWorkersStayPutWhenAGPUFrees(t *testing.T) {
	c := newCluster(t, gpu80("gpu-a", "trainer"), gpu80("gpu-b", "trainer"))
	c.arrive(trainer("z-blocker", 60))
	shared := c.arrive(trainer("w1", 50))
	if got := c.arrive(trainer("w2", 50)); got != shared {
		t.Fatalf("w2 on %q, want it sharing %q while both GPUs were taken", got, shared)
	}

	c.leave("z-blocker")
	if got := c.placed["w2"]; got.Name != shared {
		t.Fatalf("w2 moved to %q; V1 never migrates a placed worker", got.Name)
	}
	fresh := c.arrive(trainer("fresh", 50))
	if fresh == "" || c.placed["fresh"].Node != "gpu-a" {
		t.Fatalf("fresh = %+v, want it on the GPU the blocker freed", c.placed["fresh"])
	}
}

// The dev box, using the estimator's own tier outputs: 10Gi lora-tier
// workers on 2x L4. Spread first, then share, then wait for a seat.
func TestTierTableWorkloadsOnTheDevBox(t *testing.T) {
	c := newCluster(t, l4Box("box"))

	a := c.arrive(trainer("lora-1", 10))
	b := c.arrive(trainer("lora-2", 10))
	if a == b {
		t.Fatalf("both loras landed on %q while an L4 was free", a)
	}
	// The 80Gi tier's 60Gi estimate does not fit any L4, whole box or not.
	if got := c.arrive(trainer("fft-8b", 60)); got != "" {
		t.Fatalf("a 60Gi estimate landed on %q, but the box's devices are 24Gi", got)
	}
	// More loras double up on the existing claims instead.
	if got := c.arrive(trainer("lora-3", 10)); got != a {
		t.Fatalf("lora-3 landed on %q, want the fewest-workers claim %q", got, a)
	}
}

// A torchrun group needs its devices on one node and never shares them: it
// waits for a node with enough free devices, takes exactly that many, and a
// later single-device worker cannot join its claim.
func TestATorchrunGroupTakesWholeDevicesOnOneNode(t *testing.T) {
	group := Request{Role: "trainer", WorkerID: "tp4", Memory: gib(40), Devices: 4, MaxDevices: 4, HostRequestBytes: gib(40)}

	twoWide := newCluster(t, gpu80("node-a", "trainer"), gpu80("node-b", "trainer"))
	if got := twoWide.arrive(group); got != "" {
		t.Fatalf("a four-device group landed on %q in a fleet of two-device nodes", got)
	}
	if reason := twoWide.waitingReason(group); !strings.Contains(reason, "NoCapacity") {
		t.Errorf("reason = %q, want NoCapacity: no node will ever have four devices", reason)
	}

	eight := &Node{Name: "wide", DeviceCount: 8, DeviceMemoryBytes: gib(80), HostMemoryBytes: gib(1200), Roles: map[string]bool{"trainer": true}}
	c := newCluster(t, eight)
	if c.arrive(group) == "" || c.placed["tp4"].DeviceCount != 4 {
		t.Fatalf("placed = %+v, want a four-device claim", c.placed["tp4"])
	}
	second := group
	second.WorkerID = "tp4-b"
	if c.arrive(second) == "" {
		t.Fatal("the four devices left should hold a second group")
	}
	third := group
	third.WorkerID = "tp4-c"
	if got := c.arrive(third); got != "" {
		t.Fatalf("a third group landed on %q with no free devices", got)
	}
	if reason := c.waitingReason(third); !strings.Contains(reason, "WaitingForCapacity") {
		t.Errorf("reason = %q, want WaitingForCapacity: the hardware exists and is busy", reason)
	}
	// A shareable single-device worker never joins a group's claim.
	if got := c.arrive(trainer("small", 6)); got != "" {
		t.Fatalf("a single-device worker joined %q, but group claims are exclusive", got)
	}
}
