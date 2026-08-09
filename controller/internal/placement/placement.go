// Package placement is the scheduling decision: pure functions over a Request
// and a Fleet, with no Kubernetes in sight.
//
// The whole model is one sentence: a claim is a bundle of accelerators;
// cohorts within a claim take turns; members of a cohort run together.
package placement

import (
	"fmt"
	"sort"
	"time"
)

// GiB is the unit every memory figure here is reported in.
const GiB int64 = 1 << 30

// CeilGiB rounds a byte count up to whole GiB. The one rounding rule for
// every figure the scheduler reports or writes into a CEL selector.
func CeilGiB(bytes int64) int64 {
	return (bytes + GiB - 1) / GiB
}

// A cohort is the set of workers that hold a claim's accelerators at the same
// time:
//
//	memory sums *within* a cohort; cohorts *take turns*.
//
// The cohort is an opaque string the caller chooses, because only the caller
// knows what its workers share. Several lora adapters over one frozen base
// model name that base model and run at once, their device memory genuinely
// summing. A full fine-tune shares nothing, names no cohort, and is a cohort
// of one that takes its turn alone.
//
// The scheduler never interprets the string. That is the point: a new way to
// share -- two samplers over one KV cache, a trainer pinned with its sampler --
// is a caller-side decision, not a new release of this binary.
//
// The node-local timeslicer grants a turn to a cohort rather than to a single
// process, which is what makes co-residency work at runtime.

// HostMemoryHeadroom is the share of a node's allocatable memory left for the
// kubelet, the DRA driver, page cache, and the running cohort's own host-side
// allocations. Only the remainder is treated as space for parked cohorts.
const HostMemoryHeadroom = 0.15

// Switch-cost model, fitted to two measured points on 2x RTX PRO 6000
// Blackwell (driver 580.159.04) under cuda-checkpoint:
//
//	~1GiB over 1 device:    checkpoint 1.849s, restore 0.521s  -> 2.37s
//	~128GiB over 2 devices: checkpoint 21.523s, restore 8.648s -> 30.17s
//
// Two points only fit a line, and this one is not claimed to hold outside that
// range. It exists so an operator can see, before setting max-residents, that
// the same knob costs 2s on a 0.5B model and 30s on an 8B one.
const (
	switchFixedSeconds  = 2.15
	switchSecondsPerGiB = 0.2189
)

// SwitchCost is how long the timeslicer is expected to take to park a cohort
// of this size and restore the next one.
func SwitchCost(parkedBytes int64) time.Duration {
	gib := float64(parkedBytes) / float64(GiB)
	return time.Duration((switchFixedSeconds + switchSecondsPerGiB*gib) * float64(time.Second))
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
	// Roles is the set of worker roles the operator allowed on this pool.
	Roles map[string]bool
	// MaxResidents is the openrl.io/max-residents ceiling. It is a policy cap,
	// not a capacity check.
	MaxResidents int
	Product      string
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

// HostBudget is how much of this node's memory may hold parked cohorts.
//
// The operator's max-residents label caps how many workers share an allocation,
// but host memory is the other bound, and the harder one: cuda-checkpoint does
// not spill to disk, it parks a process's device memory in that process's own
// host address space. Every cohort but the one holding the accelerators is
// parked. Exceeding device memory degrades; exceeding this OOM-kills the node.
//
// Zero means the node did not report allocatable memory, in which case the
// check is skipped rather than guessed at.
func (n *Node) HostBudget() int64 {
	return int64(float64(n.HostMemoryBytes) * (1 - HostMemoryHeadroom))
}

// Claim is a ResourceClaim, plus what is already sitting on it.
//
// A claim is not partitioned by role or by workload type. Anything the claim's
// node accepts may join it; a worker that shares nothing with the residents
// simply becomes another cohort and waits its turn. Refusing to mix would only
// strand accelerators that time-slicing can already share.
type Claim struct {
	Name        string
	DeviceCount int
	// Node is where the claim was allocated, empty until DRA has decided.
	Node string
	// Residents is how many workers are assigned to it, across all cohorts.
	Residents int
	// booked is per-device bytes per cohort. A claim holds several cohorts and
	// they take turns, so this is deliberately not one total: the figures that
	// matter are the largest cohort (device memory) and everything but the
	// running one (host memory).
	booked map[string]int64
}

// Allocated reports whether the scheduler has said where this claim landed.
func (c *Claim) Allocated() bool { return c.Node != "" }

// Book accepts a placement: one more resident, its memory charged to its cohort.
func (c *Claim) Book(cohort string, perDeviceBytes int64) {
	if c.booked == nil {
		c.booked = map[string]int64{}
	}
	c.Residents++
	c.booked[cohort] += perDeviceBytes
}

// Release gives back a seat and the memory that came with it. A cohort with
// nothing left is removed rather than left at zero, so it stops taking turns.
func (c *Claim) Release(cohort string, perDeviceBytes int64) {
	c.Residents--
	if c.booked[cohort] -= perDeviceBytes; c.booked[cohort] <= 0 {
		delete(c.booked, cohort)
	}
}

// BookedIn is the per-device memory already spoken for by one cohort. This is
// what a joining member of that cohort has to fit alongside, since the cohort
// is resident all at once.
func (c *Claim) BookedIn(cohort string) int64 { return c.booked[cohort] }

// Cohorts is the set of cohorts on this claim and their per-device bytes.
func (c *Claim) Cohorts() map[string]int64 { return c.booked }

// ParkedBytesWith is the host memory this claim's parked cohorts would hold if
// one more worker of perDeviceBytes joined the named cohort.
//
// Exactly one cohort is on the accelerators and the rest are parked, so the
// worst case is the *smallest* cohort running and every other one in host RAM.
// Each parked cohort holds what it had on every device, hence DeviceCount.
func (c *Claim) ParkedBytesWith(cohort string, perDeviceBytes int64) int64 {
	var total, smallest int64
	for name, bytes := range c.booked {
		if name == cohort {
			bytes += perDeviceBytes
		}
		total += bytes
		if smallest == 0 || bytes < smallest {
			smallest = bytes
		}
	}
	if _, joined := c.booked[cohort]; !joined {
		total += perDeviceBytes
		if smallest == 0 || perDeviceBytes < smallest {
			smallest = perDeviceBytes
		}
	}
	return (total - smallest) * int64(c.DeviceCount)
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

// FreeDevices is how many of a node's accelerators no claim has taken yet.
//
// Claims DRA has not allocated do not count against any node, because nobody
// knows yet where they will land. That makes this an optimistic figure during a
// burst, which is the right direction to be wrong in: the scheduler is the
// authority on whether a claim can actually be satisfied, and a claim it cannot
// satisfy waits rather than breaking anything.
func (f *Fleet) FreeDevices(node *Node) int {
	free := node.DeviceCount
	for _, claim := range f.Claims {
		if claim.Node == node.Name {
			free -= claim.DeviceCount
		}
	}
	return free
}

// Request is one worker's needs, parsed out of its spec once.
type Request struct {
	// Role is which node pools may host this worker. It selects nodes and
	// nothing else -- in particular it does not partition claims, so a trainer
	// and a sampler on a both-roles node can share one accelerator by turns.
	Role string
	// Memory is the total accelerator memory the worker needs, across however
	// many devices it ends up on.
	Memory int64
	// Cohort is the set of workers this one may be resident alongside. Empty
	// means it shares nothing.
	Cohort string
	// WorkerID identifies the worker, and is its cohort of one when Cohort is
	// empty. Required, and required to be unique.
	WorkerID string
}

// CohortKey names the set of workers this one would hold the accelerators
// alongside. Sharing nothing is the fallback, because taking a turn is always
// safe and summing memory with strangers is not.
func (r Request) CohortKey() string {
	if r.Cohort != "" {
		return r.Cohort
	}
	return r.WorkerID
}

// DevicesOn is how many of a node's devices this workload needs, or 0 if the
// pool is too small to hold it at all.
//
// Plain ceiling division, because there is no sharding: the model is laid out
// layer by layer across whatever devices it is given, so any count will do and
// aggregate memory is the only thing that has to add up.
func (r Request) DevicesOn(n *Node) int {
	if n.DeviceMemoryBytes <= 0 {
		return 0
	}
	count := int((r.Memory + n.DeviceMemoryBytes - 1) / n.DeviceMemoryBytes)
	if count < 1 {
		count = 1
	}
	if count > n.DeviceCount {
		return 0
	}
	return count
}

// PerDeviceBytes is the workload's share of each device when spread over
// deviceCount of them.
//
// An even split, which a layer-by-layer layout only approximates: whole layers
// cannot be divided, so the last device usually runs lighter than this says.
// Erring high is the safe direction, and the estimator's safety margin absorbs
// the rest.
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

// strictestNodeValue resolves a per-node figure for a claim: the allocated
// node's value, or -- for a claim DRA has not placed yet, which could land on
// any candidate pool of its shape -- the lowest value among them. Being wrong
// in the permissive direction here means overcommitting whichever node the
// claim eventually lands on.
func strictestNodeValue(claim *Claim, fleet *Fleet, candidates map[string]int, value func(*Node) int64) int64 {
	if claim.Allocated() {
		if node := fleet.Nodes[claim.Node]; node != nil {
			return value(node)
		}
		return 0
	}
	var lowest int64
	for name, count := range candidates {
		if count != claim.DeviceCount {
			continue
		}
		if v := value(fleet.Nodes[name]); lowest == 0 || v < lowest {
			lowest = v
		}
	}
	return lowest
}

// maxResidentsFor is the operator's resident ceiling for a claim.
func maxResidentsFor(claim *Claim, fleet *Fleet, candidates map[string]int) int {
	return int(strictestNodeValue(claim, fleet, candidates, func(n *Node) int64 {
		return int64(n.MaxResidents)
	}))
}

// deviceMemoryFor is the per-device memory a claim's resident cohort has to fit
// inside.
func deviceMemoryFor(claim *Claim, fleet *Fleet, candidates map[string]int) int64 {
	return strictestNodeValue(claim, fleet, candidates, func(n *Node) int64 {
		return n.DeviceMemoryBytes
	})
}

// hostBudgetFor is the host memory a claim's parked cohorts have to fit inside.
func hostBudgetFor(claim *Claim, fleet *Fleet, candidates map[string]int) int64 {
	return strictestNodeValue(claim, fleet, candidates, func(n *Node) int64 {
		return n.HostBudget()
	})
}

// Selection is a claim this worker can join, which cohort it joins there, and
// what it costs.
type Selection struct {
	Claim          *Claim
	Cohort         string
	PerDeviceBytes int64
}

// SelectClaim is the claim this worker should join, or nil if none will have it.
//
// Preference order:
//
//  1. An allocated claim, fewest free seats first. Bin-packing, so whole
//     accelerators stay free for workers that cannot share.
//  2. A claim that exists but has not been allocated yet.
//
// Rule 2 is what keeps a burst of identical workers together. Workers created
// in the same instant all reconcile before any of their claims is allocated;
// if an unallocated claim were not joinable, each would cut its own, DRA would
// place them independently, and they would scatter across nodes with no way to
// undo it -- a claim's allocation is immutable. Joining an unallocated claim
// costs nothing: the first pod to schedule decides where it lands, and every
// other pod on the claim follows it there.
//
// The cohort is not a filter. A claim may hold as many cohorts as fit; what a
// mismatch costs is a turn, not a rejection.
func SelectClaim(req Request, fleet *Fleet) *Selection {
	candidates := candidateNodes(req, fleet)
	cohort := req.CohortKey()

	var best *scored

	for _, claim := range fleet.Claims {
		if claim.Allocated() {
			node := fleet.Nodes[claim.Node]
			if node == nil || !node.Accepts(req.Role) || req.DevicesOn(node) != claim.DeviceCount {
				continue
			}
		} else if !anyNodeTakes(candidates, claim.DeviceCount) {
			// Nothing this workload fits on would produce a claim this shape.
			continue
		}

		seats := maxResidentsFor(claim, fleet, candidates) - claim.Residents
		if seats <= 0 {
			continue
		}

		// The cohort is resident all at once, so its members sum against the
		// device. Cohorts this worker is not joining do not: they are parked.
		perDevice := req.PerDeviceBytes(claim.DeviceCount)
		deviceMemory := deviceMemoryFor(claim, fleet, candidates)
		if deviceMemory == 0 || claim.BookedIn(cohort)+perDevice > deviceMemory {
			continue
		}

		// Everything not running is in host RAM. A node that did not report its
		// allocatable memory reports a zero budget; skip the check rather than
		// refuse every claim on it.
		if budget := hostBudgetFor(claim, fleet, candidates); budget > 0 && claim.ParkedBytesWith(cohort, perDevice) > budget {
			continue
		}

		pending := 0
		if !claim.Allocated() {
			pending = 1
		}
		cand := &scored{claim: claim, perDev: perDevice, pending: pending, freeSeat: seats}
		if best == nil || cand.before(best) {
			best = cand
		}
	}

	if best == nil {
		return nil
	}
	return &Selection{Claim: best.claim, Cohort: cohort, PerDeviceBytes: best.perDev}
}

// scored is a joinable claim and the figures it is ranked by.
type scored struct {
	claim    *Claim
	perDev   int64
	pending  int // 0 for allocated claims, 1 for not-yet-allocated ones
	freeSeat int
}

// before orders candidate claims: allocated before pending, then fewest free
// seats, then by name so the choice is stable across reconciles.
func (a *scored) before(b *scored) bool {
	if a.pending != b.pending {
		return a.pending < b.pending
	}
	if a.freeSeat != b.freeSeat {
		return a.freeSeat < b.freeSeat
	}
	return a.claim.Name < b.claim.Name
}

func anyNodeTakes(candidates map[string]int, deviceCount int) bool {
	for _, count := range candidates {
		if count == deviceCount {
			return true
		}
	}
	return false
}

// Pool is the node a new claim is sized against and how wide it would be.
type Pool struct {
	Node        *Node
	DeviceCount int
}

// ChoosePool picks the pool a new claim is sized against, or nil if no pool
// has room for one.
//
// Best fit by wasted memory, among pools with enough accelerators still
// unclaimed. This only sizes the claim: the scheduler picks where it actually
// lands, which may be another node of the same shape.
func ChoosePool(req Request, fleet *Fleet) *Pool {
	var best *Pool
	var bestWaste int64
	names := make([]string, 0, len(fleet.Nodes))
	for name := range fleet.Nodes {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		node := fleet.Nodes[name]
		if !node.Accepts(req.Role) {
			continue
		}
		count := req.DevicesOn(node)
		if count == 0 || count > fleet.FreeDevices(node) {
			continue
		}
		waste := int64(count)*node.DeviceMemoryBytes - req.Memory
		if best == nil || waste < bestWaste || (waste == bestWaste && count < best.DeviceCount) {
			best, bestWaste = &Pool{Node: node, DeviceCount: count}, waste
		}
	}
	return best
}

// Explain says why this worker is not running: the fleet is full, or too
// small, or its host memory cannot carry another parked worker.
//
// detail is the caller's own words, kept because they name the constraint that
// actually failed.
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
		reason = fmt.Sprintf("NoCapacity: needs %dGi in total; largest pool offers %s",
			CeilGiB(req.Memory), biggest.Describe())
	}

	if detail != "" {
		reason += ". " + detail
	}
	return truncate(reason, 1024)
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
