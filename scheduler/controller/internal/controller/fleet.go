package controller

import (
	"context"
	"fmt"
	"sort"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

// Labels the controller stamps on the objects it owns, and reads back to
// rebuild its own state after a restart.
const (
	LabelManaged = "openrl.io/managed"
	LabelRole    = "openrl.io/role"
	LabelClaim   = "openrl.io/claim"
	LabelWorker  = "openrl.io/worker"
)

// Node labels the operator sets to opt a pool in. Policy, not hardware: the
// DRA driver's ResourceSlices report what the devices actually are.
//
// Opting a node in means opting it in exclusively: the controller counts a
// device as free unless one of its own claims holds it, so GPU workloads it
// does not manage on an enabled node would be invisible to placement.
const (
	NodeLabelEnabled = "openrl.io/enabled"
	NodeLabelTrainer = "openrl.io/trainer"
	NodeLabelSampler = "openrl.io/sampler"
)

var nodeRoleLabel = map[openrlv1alpha1.WorkerRole]string{
	openrlv1alpha1.RoleTrainer: NodeLabelTrainer,
	openrlv1alpha1.RoleSampler: NodeLabelSampler,
}

// readFleet folds ResourceSlices, node labels, claims and ledger seats into
// one snapshot. Claims and ledgers are read through fleetReader so a seat
// booked one reconcile ago is visible immediately; nodes, slices and foreign
// pods come from the informer cache.
func (r *WorkloadReconciler) readFleet(ctx context.Context) (*placement.Fleet, error) {
	var nodes corev1.NodeList
	if err := r.List(ctx, &nodes, client.MatchingLabels{NodeLabelEnabled: "true"}); err != nil {
		return nil, fmt.Errorf("list nodes: %w", err)
	}

	var slices resourcev1.ResourceSliceList
	if err := r.List(ctx, &slices); err != nil {
		return nil, fmt.Errorf("list resourceslices: %w", err)
	}

	fleet := placement.NewFleet()
	fleet.Nodes = r.poolsFrom(ctx, slices.Items, nodes.Items)

	// kube-scheduler admits a pod against allocatable minus every pod already
	// on the node, not just the ones this controller placed. Reserving the
	// rest here keeps the host-memory fit honest, so a shared seat is never
	// booked on a node whose remaining memory kube-scheduler will refuse.
	var pods corev1.PodList
	if err := r.List(ctx, &pods); err != nil {
		return nil, fmt.Errorf("list pods: %w", err)
	}
	for node, reserved := range foreignHostBytes(pods.Items, r.Namespace) {
		if pool, ok := fleet.Nodes[node]; ok {
			pool.HostReservedBytes = reserved
		}
	}

	var claims resourcev1.ResourceClaimList
	if err := r.fleetReader().List(ctx, &claims, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return nil, fmt.Errorf("list resourceclaims: %w", err)
	}
	for i := range claims.Items {
		// A terminating claim is not placeable state: DRA's finalizer can
		// hold it for a while, and pointing anything at it only wedges.
		if claims.Items[i].DeletionTimestamp != nil {
			continue
		}
		c := claimFrom(&claims.Items[i])
		fleet.Claims[c.Name] = c
	}

	// Seats are the one occupancy record; no workload is ever read here.
	var ledgers openrlv1alpha1.ClaimLedgerList
	if err := r.fleetReader().List(ctx, &ledgers, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return nil, fmt.Errorf("list claim ledgers: %w", err)
	}
	for i := range ledgers.Items {
		ledger := &ledgers.Items[i]
		claim, ok := fleet.Claims[ledger.Spec.ClaimName]
		if !ok {
			continue
		}
		for _, seat := range ledger.Spec.Seats {
			claim.Book(seat.Workload, seat.OwnerID, seat.HostRequest.Value(), !seat.Exclusive)
		}
	}
	return fleet, nil
}

// poolsFrom merges what the driver publishes with what the operator allowed.
// Devices accumulate across slices; where memory differs the smallest wins,
// because the fit must hold for whichever devices DRA picks.
func (r *WorkloadReconciler) poolsFrom(ctx context.Context, slices []resourcev1.ResourceSlice, nodes []corev1.Node) map[string]*placement.Node {
	logger := log.FromContext(ctx)

	devices := map[string]*placement.Node{}
	for _, i := range latestCompletePools(ctx, slices, r.DeviceDriver) {
		spec := slices[i].Spec
		name := *spec.NodeName
		for j := range spec.Devices {
			device := spec.Devices[j]
			capacity, ok := device.Capacity["memory"]
			if !ok {
				continue
			}
			memory := capacity.Value.Value()
			pool, seen := devices[name]
			if !seen {
				product := ""
				if attr, ok := device.Attributes["productName"]; ok && attr.StringValue != nil {
					product = *attr.StringValue
				}
				devices[name] = &placement.Node{Name: name, DeviceCount: 1, DeviceMemoryBytes: memory, Product: product}
				continue
			}
			pool.DeviceCount++
			pool.DeviceMemoryBytes = min(pool.DeviceMemoryBytes, memory)
		}
	}

	pools := map[string]*placement.Node{}
	for i := range nodes {
		node := &nodes[i]
		pool, ok := devices[node.Name]
		if !ok {
			logger.Info("node is enabled but no ResourceSlice from this driver describes it; skipping it for placement",
				"node", node.Name, "driver", r.DeviceDriver)
			continue
		}
		// No role labels means both roles. A label set to "false" denies
		// that role and does not fall back to both. This mirrors the pod
		// affinity, so placement never offers a node the pod cannot land on.
		pool.Roles = map[string]bool{}
		labeled := false
		for role, label := range nodeRoleLabel {
			value, present := node.Labels[label]
			labeled = labeled || present
			if value == "true" {
				pool.Roles[string(role)] = true
			}
		}
		if !labeled {
			for role := range nodeRoleLabel {
				pool.Roles[string(role)] = true
			}
		}
		pool.HostMemoryBytes = node.Status.Allocatable.Memory().Value()
		if pool.HostMemoryBytes == 0 {
			logger.Info("node reports no allocatable memory; parked-worker capacity cannot be checked here",
				"node", node.Name)
		}
		pools[node.Name] = pool
	}
	return pools
}

// foreignHostBytes sums, per node, the memory requests of pods this
// controller did not place: anything outside the worker namespace, and
// anything in it not bound to one of our claims. Worker pods are already on
// the ledger as seats. Finished pods hold nothing. The request follows
// kube-scheduler's arithmetic: the larger of the containers' sum and any
// single init container, plus the pod overhead.
func foreignHostBytes(pods []corev1.Pod, namespace string) map[string]int64 {
	reserved := map[string]int64{}
	for i := range pods {
		pod := &pods[i]
		if pod.Spec.NodeName == "" || pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			continue
		}
		if pod.Namespace == namespace && pod.Labels[LabelClaim] != "" {
			continue
		}
		reserved[pod.Spec.NodeName] += podHostRequestBytes(pod)
	}
	return reserved
}

func podHostRequestBytes(pod *corev1.Pod) int64 {
	var containers, init int64
	for i := range pod.Spec.Containers {
		containers += pod.Spec.Containers[i].Resources.Requests.Memory().Value()
	}
	for i := range pod.Spec.InitContainers {
		init = max(init, pod.Spec.InitContainers[i].Resources.Requests.Memory().Value())
	}
	total := max(containers, init)
	if pod.Spec.Overhead != nil {
		total += pod.Spec.Overhead.Memory().Value()
	}
	return total
}

// latestCompletePools filters slices to the latest complete generation of
// each (node, pool) -- what the ResourceSlice contract requires of consumers.
// During a driver update, mixing generations double-counts devices and a
// partial generation under-counts them; an incomplete pool is skipped and
// picked up on a later reconcile.
func latestCompletePools(ctx context.Context, slices []resourcev1.ResourceSlice, driver string) []int {
	type poolKey struct{ node, pool string }
	byPool := map[poolKey][]int{}
	for i := range slices {
		spec := slices[i].Spec
		if spec.NodeName == nil || *spec.NodeName == "" || spec.Driver != driver {
			continue
		}
		key := poolKey{*spec.NodeName, spec.Pool.Name}
		byPool[key] = append(byPool[key], i)
	}

	var keep []int
	for key, indices := range byPool {
		latest := int64(-1)
		for _, i := range indices {
			if g := slices[i].Spec.Pool.Generation; g > latest {
				latest = g
			}
		}
		var current []int
		for _, i := range indices {
			if slices[i].Spec.Pool.Generation == latest {
				current = append(current, i)
			}
		}
		if int64(len(current)) != slices[current[0]].Spec.Pool.ResourceSliceCount {
			log.FromContext(ctx).Info("skipping incomplete ResourceSlice pool",
				"node", key.node, "pool", key.pool, "have", len(current), "want", slices[current[0]].Spec.Pool.ResourceSliceCount)
			continue
		}
		keep = append(keep, current...)
	}
	sort.Ints(keep)
	return keep
}

// claimFrom reads a managed claim back into the fleet: identity from
// metadata, shape from DRA's allocation. An unallocated claim has no shape
// yet -- the tiers were alternatives, and which one holds is DRA's answer.
func claimFrom(claim *resourcev1.ResourceClaim) *placement.Claim {
	c := &placement.Claim{
		Name: claim.Name,
		Node: allocatedNode(claim),
	}
	if claim.Status.Allocation != nil {
		c.DeviceCount = len(claim.Status.Allocation.Devices.Results)
	}
	return c
}

// allocatedNode is the node a claim was allocated to, or "" if DRA has not
// decided. DRA reports placement as a node selector; a GPU allocation pins to
// exactly one hostname.
func allocatedNode(claim *resourcev1.ResourceClaim) string {
	if claim.Status.Allocation == nil || claim.Status.Allocation.NodeSelector == nil {
		return ""
	}
	for _, term := range claim.Status.Allocation.NodeSelector.NodeSelectorTerms {
		for _, expr := range term.MatchExpressions {
			if isHostnameKey(expr.Key) && len(expr.Values) > 0 {
				return expr.Values[0]
			}
		}
		for _, field := range term.MatchFields {
			if isHostnameKey(field.Key) && len(field.Values) > 0 {
				return field.Values[0]
			}
		}
	}
	return ""
}

func isHostnameKey(key string) bool {
	return key == corev1.LabelHostname || key == "metadata.name"
}

// requestFrom is the placement Request an Workload spec is asking for.
// Validation is the CRD schema's job.
func requestFrom(worker *openrlv1alpha1.Workload) placement.Request {
	spec := worker.Spec
	return placement.Request{
		Shareable: !spec.Exclusive,
		Role:      string(spec.Role),
		Memory:    spec.Accelerator.Memory.Value(),
		// Raw: the spec calls the owner ID opaque, and sanitizing here would
		// merge distinct owners ("A/B" and "a-b") into one fairness slot.
		// Labels sanitize at the stamping site; env vars carry this exactly.
		Owner: spec.OwnerID,
		// The CR name: the one identity Kubernetes already guarantees unique.
		// Model id is model configuration, not object identity.
		WorkerID: worker.Name,
		// Every current runtime drives one device -- the SingleGPU mode;
		// a wider claim arrives as a new accelerator mode.
		MaxDevices:       1,
		HostRequestBytes: hostRequestBytes(worker),
	}
}

// hostRequestBytes is the worker container's memory request from the inline
// template: the host memory this pod will ask its node for. Zero when the
// template requests none; the host-memory admission check is then skipped
// for this worker's share.
func hostRequestBytes(worker *openrlv1alpha1.Workload) int64 {
	for i := range worker.Spec.Template.Spec.Containers {
		c := &worker.Spec.Template.Spec.Containers[i]
		if c.Name != workerContainerName(worker) {
			continue
		}
		if quantity, ok := c.Resources.Requests[corev1.ResourceMemory]; ok {
			return quantity.Value()
		}
	}
	return 0
}

// gibQuantity renders a byte count as the Gi string the CRD reports.
func gibQuantity(bytes int64) string {
	return resource.NewQuantity(placement.CeilGiB(bytes)*placement.GiB, resource.BinarySI).String()
}
