package controller

import (
	"context"
	"fmt"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// Labels the controller stamps on the objects it owns, and reads back to
// rebuild its own state after a restart.
const (
	LabelManaged    = "openrl.io/managed"
	LabelRole       = "openrl.io/role"
	LabelAccelCount = "openrl.io/accelerator-count"
	LabelClaim      = "openrl.io/claim"
	LabelWorker     = "openrl.io/worker"
)

// Node labels the operator sets to opt a pool in. These express policy, not
// hardware: openrl.io/trainer=true means the controller may place trainers
// here, never that the node holds any particular accelerator. The DRA driver
// reports what the devices actually are.
const (
	NodeLabelEnabled      = "openrl.io/enabled"
	NodeLabelMaxResidents = "openrl.io/max-residents"
	NodeLabelTrainer      = "openrl.io/trainer"
	NodeLabelSampler      = "openrl.io/sampler"
)

var nodeRoleLabel = map[openrlv1alpha1.WorkerRole]string{
	openrlv1alpha1.RoleTrainer: NodeLabelTrainer,
	openrlv1alpha1.RoleSampler: NodeLabelSampler,
}

// readFleet folds ResourceSlices, node labels, managed ResourceClaims and the
// workers already assigned to them into one picture to decide against.
func (r *OpenRLWorkerReconciler) readFleet(ctx context.Context, workers []openrlv1alpha1.OpenRLWorker) (*placement.Fleet, error) {
	var nodes corev1.NodeList
	if err := r.List(ctx, &nodes, client.MatchingLabels{NodeLabelEnabled: "true"}); err != nil {
		return nil, fmt.Errorf("list nodes: %w", err)
	}

	var slices resourcev1.ResourceSliceList
	if err := r.List(ctx, &slices); err != nil {
		// Without slices there are no devices, so a claim cut now could never
		// be satisfied. Placing nothing is the honest outcome.
		return nil, fmt.Errorf("list resourceslices: %w", err)
	}

	fleet := placement.NewFleet()
	fleet.Nodes = r.poolsFrom(ctx, slices.Items, nodes.Items)

	// Claims come through the consistent reader: a claim created for the
	// previous worker in a burst must be joinable by this one, and the informer
	// cache does not promise to have caught up yet.
	var claims resourcev1.ResourceClaimList
	if err := r.fleetReader().List(ctx, &claims, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return nil, fmt.Errorf("list resourceclaims: %w", err)
	}
	for i := range claims.Items {
		if c := r.claimFrom(ctx, &claims.Items[i]); c != nil {
			fleet.Claims[c.Name] = c
		}
	}

	// Residents come from the workers themselves, not from their pods: the
	// worker's status is the controller's own record of the decision, so it is
	// correct in the window before a pod exists and after one has restarted.
	for i := range workers {
		bookWorker(fleet, &workers[i])
	}
	return fleet, nil
}

// poolsFrom merges what the driver publishes with what the operator allowed.
//
// A node may be described by more than one slice, so devices accumulate. Where
// they differ the smallest memory wins: the controller commits to a device
// count before the scheduler picks which devices, so the fit has to hold for
// whichever ones it gets.
func (r *OpenRLWorkerReconciler) poolsFrom(ctx context.Context, slices []resourcev1.ResourceSlice, nodes []corev1.Node) map[string]*placement.Node {
	logger := log.FromContext(ctx)

	devices := map[string]*placement.Node{}
	for i := range slices {
		spec := slices[i].Spec
		if spec.NodeName == nil || *spec.NodeName == "" || spec.Driver != r.DeviceDriver {
			continue
		}
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
		pool.Roles = map[string]bool{}
		for role, label := range nodeRoleLabel {
			if node.Labels[label] == "true" {
				pool.Roles[string(role)] = true
			}
		}
		pool.MaxResidents = max(1, labelInt(ctx, node.Labels, NodeLabelMaxResidents, 1, "node "+node.Name))
		pool.HostMemoryBytes = node.Status.Allocatable.Memory().Value()
		if pool.HostMemoryBytes == 0 {
			logger.Info("node reports no allocatable memory; parked-worker capacity cannot be checked here",
				"node", node.Name)
		}
		pools[node.Name] = pool
	}
	return pools
}

// claimFrom reads back the shape the controller stamped on a claim it created.
func (r *OpenRLWorkerReconciler) claimFrom(ctx context.Context, claim *resourcev1.ResourceClaim) *placement.Claim {
	count := labelInt(ctx, claim.Labels, LabelAccelCount, 0, "claim "+claim.Name)
	if count < 1 {
		log.FromContext(ctx).Info("skipping claim with unusable accelerator-count label", "claim", claim.Name)
		return nil
	}
	return &placement.Claim{
		Name:        claim.Name,
		DeviceCount: count,
		Node:        allocatedNode(claim),
	}
}

// bookWorker charges a worker's assignment against the cohort it joined on the
// claim it names. The memory is always re-derived from the spec: the status
// carries a Gi-rounded display string, and one derivation is one thing to keep
// correct. The cohort is likewise re-derived rather than read back from a
// label, so the claim's occupancy always reflects the current cohort rule.
func bookWorker(fleet *placement.Fleet, worker *openrlv1alpha1.OpenRLWorker) {
	if claim, ok := fleet.Claims[worker.Status.ClaimName]; ok {
		request := requestFrom(worker)
		claim.Book(request.CohortKey(), request.PerDeviceBytes(claim.DeviceCount))
	}
}

// allocatedNode is the node a claim was allocated to, or "" if DRA has not
// placed it yet.
//
// DRA reports placement as a node selector, since in principle a claim can be
// satisfiable on several nodes. A GPU allocation pins to exactly one hostname.
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

// requestFrom is the placement Request an OpenRLWorker spec is asking for.
//
// No validation here: the CRD schema owns it.
func requestFrom(worker *openrlv1alpha1.OpenRLWorker) placement.Request {
	spec := worker.Spec
	return placement.Request{
		Role:   string(spec.Role),
		Memory: spec.Memory.Value(),
		Cohort: sanitizeLabel(spec.Cohort),
		// The model id is the worker's identity everywhere else -- it names its
		// pod and its time-slice job -- so it names its cohort of one too.
		WorkerID: sanitizeLabel(spec.ModelID),
	}
}

// labelInt reads a non-negative integer label, falling back to a default if it
// is missing or nonsense.
func labelInt(ctx context.Context, labels map[string]string, key string, fallback int, subject string) int {
	raw, ok := labels[key]
	if !ok {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value < 0 {
		log.FromContext(ctx).Info("ignoring unparseable label", "subject", subject, "label", key, "value", raw, "using", fallback)
		return fallback
	}
	return value
}

// gibQuantity renders a byte count as the Gi string the CRD reports.
func gibQuantity(bytes int64) string {
	return resource.NewQuantity(placement.CeilGiB(bytes)*placement.GiB, resource.BinarySI).String()
}
