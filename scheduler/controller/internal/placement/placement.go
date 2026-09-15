// Package placement is the scheduling decision: pure functions over a Request
// and a Fleet, no Kubernetes imports. A claim is a bundle of accelerators;
// several workers may be assigned to it; exactly one is resident at a time.
//
// Placement never surveys free capacity. A new worker's claim states an
// ordered set of acceptable device shapes (Tiers) and DRA decides whether one
// is free -- kube-scheduler's allocation cycle is the mutex, not this
// package. What remains here is the shape catalog (Tiers), the seat decision
// (SelectClaim) -- a Strategy orders it before or after the dedicated claim
// -- and the explanation when neither move can help (Explain).
package placement

import (
	"fmt"
)

// GiB is the unit every memory figure here is reported in.
const GiB int64 = 1 << 30

// Strategy orders the two placement moves. BinPack seats a worker on an
// existing ledger first and cuts a claim only when none can hold it; Spread
// cuts a dedicated claim first and shares when the cluster says no.
// The zero value reads as binpack.
type Strategy string

const (
	StrategySpread  Strategy = "spread"
	StrategyBinPack Strategy = "binpack"
)

// ParseStrategy validates an operator-supplied strategy name; empty means
// binpack.
func ParseStrategy(name string) (Strategy, error) {
	switch Strategy(name) {
	case "", StrategyBinPack:
		return StrategyBinPack, nil
	case StrategySpread:
		return Strategy(name), nil
	}
	return "", fmt.Errorf("unknown placement strategy %q: want %q or %q", name, StrategySpread, StrategyBinPack)
}

// CeilGiB rounds a byte count up to whole GiB. Used for reported figures,
// tier names and CEL ceilings; CEL floors stay in exact bytes.
func CeilGiB(bytes int64) int64 {
	return (bytes + GiB - 1) / GiB
}

// Node is one accelerator pool: hardware from the driver's ResourceSlice,
// policy from the operator's node labels.
type Node struct {
	Name string
	// DeviceCount and DeviceMemoryBytes come from the DRA driver.
	DeviceCount       int
	DeviceMemoryBytes int64
	// HostMemoryBytes is the node's allocatable memory, which bounds how many
	// workers can be parked here at once.
	HostMemoryBytes int64
	// HostReservedBytes is the memory requested by pods on the node that this
	// scheduler did not place (exporters, agents, system pods). kube-scheduler
	// counts them against the same allocatable pool, so a fit that ignores
	// them books a seat the pod can never take.
	HostReservedBytes int64
	// Roles is the set of worker roles the operator allowed on this pool.
	Roles   map[string]bool
	Product string
}

// Accepts reports whether the operator allowed this role on this pool.
func (n *Node) Accepts(role string) bool { return n.Roles[role] }

// Describe renders the pool's hardware for an error message.
func (n *Node) Describe() string {
	hardware := fmt.Sprintf("%dGi x %d", n.DeviceMemoryBytes/GiB, n.DeviceCount)
	if n.Product == "" {
		return hardware
	}
	return hardware + " " + n.Product
}

// booking is one assigned worker's footprint on a claim: its owner (the
// fairness unit) and its pod's host-memory request. No accelerator figure:
// only one worker is ever resident, so nothing is summed against the device.
type booking struct {
	owner     string
	hostBytes int64
	shareable bool
}

// Claim is a ResourceClaim, plus what is already sitting on it. Claims are
// shareable only when every seated worker can suspend between GPU turns.
type Claim struct {
	Name string
	// DeviceCount is how many devices DRA allocated, 0 until it has.
	DeviceCount int
	// Node is where the claim was allocated, empty until DRA has decided.
	Node string
	// booked is each assigned worker's footprint -- deliberately never one
	// total, because nothing is ever summed against the device.
	booked map[string]booking
}

// Allocated reports whether the scheduler has said where this claim landed.
func (c *Claim) Allocated() bool { return c.Node != "" }

// Workers is how many workers are assigned to this claim.
func (c *Claim) Workers() int { return len(c.booked) }

// Book accepts a placement: one more assigned worker and its footprint.
func (c *Claim) Book(workerID, owner string, hostBytes int64, shareable bool) {
	if c.booked == nil {
		c.booked = map[string]booking{}
	}
	c.booked[workerID] = booking{owner: owner, hostBytes: hostBytes, shareable: shareable}
}

// Shareable requires every existing worker to participate in time slicing.
// An empty snapshot is not shareable.
func (c *Claim) Shareable() bool {
	if len(c.booked) == 0 {
		return false
	}
	for _, b := range c.booked {
		if !b.shareable {
			return false
		}
	}
	return true
}

// Release gives back a worker's seat and the memory that came with it.
func (c *Claim) Release(workerID string) {
	delete(c.booked, workerID)
}

// Booked reports whether this worker holds a seat on the claim.
func (c *Claim) Booked(workerID string) bool {
	_, ok := c.booked[workerID]
	return ok
}

// Owners is how many distinct fairness units are assigned to this claim.
func (c *Claim) Owners() int {
	owners := map[string]bool{}
	for _, b := range c.booked {
		owners[b.owner] = true
	}
	return len(owners)
}

// Fleet is everything placement decides against.
type Fleet struct {
	Nodes  map[string]*Node
	Claims map[string]*Claim
}

// NewFleet returns an empty Fleet.
func NewFleet() *Fleet {
	return &Fleet{Nodes: map[string]*Node{}, Claims: map[string]*Claim{}}
}

// NodeHostBytes is the host memory every worker already assigned to this
// node's claims will request from it. Summed across claims, because the
// node's allocatable memory is one pool however many claims sit on it.
func (f *Fleet) NodeHostBytes(node *Node) int64 {
	total := node.HostReservedBytes
	for _, claim := range f.Claims {
		if claim.Node != node.Name {
			continue
		}
		for _, b := range claim.booked {
			total += b.hostBytes
		}
	}
	return total
}

// Request is one worker's needs, parsed out of its spec once.
type Request struct {
	// Shareable means the worker can suspend between GPU turns.
	Shareable bool
	// Role selects node pools and nothing else; it does not partition claims.
	Role string
	// Memory is the total accelerator memory the worker needs, across however
	// many devices it ends up on.
	Memory int64
	// Owner is the runtime fairness unit. Placement never reads it; the
	// timeslicer does.
	Owner string
	// WorkerID identifies the worker. Required, and required to be unique.
	WorkerID string
	// MaxDevices is the widest claim the runtime can drive; placement never
	// sizes one wider. Zero means one: no shipped runtime is multi-device,
	// and a wider claim is memory the pod can see but the process won't use.
	MaxDevices int
	// HostRequestBytes is the pod template's memory request: what this
	// worker's pod asks the node for, resident or parked.
	HostRequestBytes int64
}

// OwnerKey is the fairness unit the timeslicer serves this worker under; a
// worker naming no owner is an owner of one.
func (r Request) OwnerKey() string {
	if r.Owner != "" {
		return r.Owner
	}
	return r.WorkerID
}

// DevicesOn is how many of a node's devices this workload needs, or 0 if the
// pool cannot hold it within the shape the runtime declared. Ceiling
// division bounded by MaxDevices: a pool whose devices are too small to
// satisfy the worker within that many devices is ineligible, never padded
// out with devices the process cannot drive.
func (r Request) DevicesOn(n *Node) int {
	if n.DeviceMemoryBytes <= 0 {
		return 0
	}
	count := devicesFor(r.Memory, n.DeviceMemoryBytes)
	if count > r.widest() || count > n.DeviceCount {
		return 0
	}
	return count
}

// widest is the most devices the runtime declared it can drive; zero means one.
func (r Request) widest() int { return max(1, r.MaxDevices) }

// devicesFor is how many devices of one size hold the memory: ceiling
// division, at least one.
func devicesFor(memory, deviceBytes int64) int {
	return max(1, int((memory+deviceBytes-1)/deviceBytes))
}

// PerDeviceBytes is the workload's share of each device when spread over
// deviceCount of them. An even split that a layer-wise layout only
// approximates; erring high is the safe direction.
func (r Request) PerDeviceBytes(deviceCount int) int64 {
	if deviceCount < 1 {
		panic(fmt.Sprintf("deviceCount must be >= 1, got %d", deviceCount))
	}
	return (r.Memory + int64(deviceCount) - 1) / int64(deviceCount)
}

// candidateNodes is every pool that accepts the role and fits the workload,
// with the device count it would take there.
func candidateNodes(req Request, fleet *Fleet) map[string]int {
	fits := map[string]int{}
	for name, node := range fleet.Nodes {
		if !node.Accepts(req.Role) {
			continue
		}
		if count := req.DevicesOn(node); count > 0 {
			fits[name] = count
		}
	}
	return fits
}

// SelectClaim picks the allocated claim a worker should join, or nil:
// single-device only, role and host-memory checked -- host memory is the one
// ceiling on how many workers may park on a node -- preferring fewest owners,
// then fewest workers, then name. Advisory -- the ledger's CAS booking is what
// makes it stick.
func SelectClaim(req Request, fleet *Fleet) *Claim {
	if !req.Shareable {
		return nil
	}
	// One pass over the fleet up front: summing bookings per candidate would
	// rescan every claim for every claim, and SelectClaim runs on every
	// reconcile of an unplaced worker under BinPack.
	hostByNode := map[string]int64{}
	for name, node := range fleet.Nodes {
		hostByNode[name] = node.HostReservedBytes
	}
	for _, claim := range fleet.Claims {
		if claim.Node == "" {
			continue
		}
		for _, b := range claim.booked {
			hostByNode[claim.Node] += b.hostBytes
		}
	}

	var best *Claim
	for _, claim := range fleet.Claims {
		node := fleet.Nodes[claim.Node]
		if !claim.Shareable() || !claim.Allocated() || node == nil || !node.Accepts(req.Role) {
			continue
		}
		if claim.DeviceCount != 1 || req.DevicesOn(node) != 1 {
			continue
		}
		if req.Memory > node.DeviceMemoryBytes {
			continue
		}
		// Node-wide: every pod on the node draws from one allocatable pool.
		// A node reporting no memory skips the check rather than refusing all.
		if node.HostMemoryBytes > 0 && hostByNode[node.Name]+req.HostRequestBytes > node.HostMemoryBytes {
			continue
		}
		if best == nil || claimLess(claim, best) {
			best = claim
		}
	}
	return best
}

// claimLess orders eligible claims: fewer assigned owners first (joining the
// least-contended fairness pool), then fewer workers, then name.
func claimLess(a, b *Claim) bool {
	if a.Owners() != b.Owners() {
		return a.Owners() < b.Owners()
	}
	if a.Workers() != b.Workers() {
		return a.Workers() < b.Workers()
	}
	return a.Name < b.Name
}

// Explain says why this worker is not running. "Busy, retry" and "too small,
// never" are deliberately different answers; detail carries the caller's own
// words about what failed.
func Explain(req Request, fleet *Fleet, detail string) string {
	var pools []*Node
	for _, node := range fleet.Nodes {
		if node.Accepts(req.Role) {
			pools = append(pools, node)
		}
	}

	var reason string
	switch {
	case len(pools) == 0:
		reason = fmt.Sprintf("NoCapacity: no enabled node accepts %s workers", req.Role)
	case len(candidateNodes(req, fleet)) > 0:
		// The hardware exists; it is busy. Retrying is the right move.
		reason = "WaitingForCapacity: a pool fits this workload but none has a free seat or a free accelerator"
	default:
		biggest := pools[0]
		for _, node := range pools[1:] {
			if int64(node.DeviceCount)*node.DeviceMemoryBytes > int64(biggest.DeviceCount)*biggest.DeviceMemoryBytes {
				biggest = node
			}
		}
		reason = fmt.Sprintf("NoCapacity: needs %dGi across at most %d device(s); largest pool offers %s",
			CeilGiB(req.Memory), req.widest(), biggest.Describe())
	}

	if detail != "" {
		reason += ". " + detail
	}
	if len(reason) > 1024 {
		reason = reason[:1024]
	}
	return reason
}
