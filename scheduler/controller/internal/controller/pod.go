package controller

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

// Time-slicer contract, mirrored from src/accel_timeslicer/workload.py. A pod
// label so other pods can discover the workload, an env var so the process
// itself knows which ledger and owner it belongs to. All of them must agree.
const (
	timeSliceEnabledLabel = "accel-timeslicer"
	timeSliceGroupLabel   = "timeslice.io/group"
	timeSliceOwnerLabel   = "timeslice.io/owner"
	timeSliceJobIDLabel   = "timeslice.io/job-id"
	timeSliceJobIDEnv     = "OPEN_RL_TIME_SLICE_JOB_ID"
	timeSliceGroupEnv     = "OPEN_RL_TIME_SLICE_GROUP"
	timeSliceOwnerEnv     = "OPEN_RL_TIME_SLICE_OWNER"
	claimLedgerEnv        = "OPEN_RL_CLAIM_LEDGER"
	assignmentIDEnv       = "OPEN_RL_ASSIGNMENT_ID"
)

// The name the pod and the claim agree to call the allocation.
const podClaimName = "gpu"

// labelUnsafe matches everything a DNS-1123 label value may not contain.
var labelUnsafe = regexp.MustCompile(`[^a-z0-9-]+`)

// sanitizeLabel reduces an arbitrary id to something usable as a label value.
// Empty in, empty out.
func sanitizeLabel(value string) string {
	cleaned := strings.Trim(labelUnsafe.ReplaceAllString(strings.ToLower(value), "-"), "-")
	if len(cleaned) > 63 {
		cleaned = strings.TrimRight(cleaned[:63], "-")
	}
	return cleaned
}

// workerPodName derives the pod from the CR name -- the one identity
// Kubernetes already guarantees unique. A truncated name keeps a hash of the
// full one so two long names cannot collide.
func workerPodName(worker *openrlv1alpha1.Workload) string {
	name := "orw-" + worker.Name
	if len(name) <= 253 {
		return name
	}
	sum := sha256.Sum256([]byte(name))
	return name[:245] + "-" + hex.EncodeToString(sum[:])[:7]
}

// claimNameFor is UID-derived: unique per incarnation, stable across
// reconciles, truncated to survive as a label value.
func claimNameFor(worker *openrlv1alpha1.Workload) string {
	name := worker.Name
	if len(name) > 48 {
		name = strings.TrimRight(name[:48], "-.")
	}
	uid := string(worker.UID)
	if uid == "" {
		// Objects always carry a UID in a real cluster; bare fixtures don't.
		return "claim-" + name
	}
	if len(uid) > 8 {
		uid = uid[:8]
	}
	return "claim-" + name + "-" + uid
}

// buildClaim renders the tiers as ordered firstAvailable alternatives, each
// bounded by a CEL memory floor and ceiling. The floor is written in exact
// bytes because devices report fractional sizes (an 80GB A100 is ~79.65Gi)
// and a floor rounded up to whole GiB could exceed the device. The ceiling
// rounds up, which admits nothing new. There is no node selector; that is
// kube-scheduler's call.
//
// With a DeviceMemoryTable the bounds are written against the table's
// attribute instead, since such devices carry no capacity to compare. Under
// WholeNodeClaims each tier asks for all matching devices on the node.
func (r *WorkloadReconciler) buildClaim(claimName string, tiers []placement.Tier) *resourcev1.ResourceClaim {
	subrequests := make([]resourcev1.DeviceSubRequest, len(tiers))
	for i, tier := range tiers {
		subrequests[i] = resourcev1.DeviceSubRequest{
			Name:            tier.Name,
			DeviceClassName: r.DeviceClass,
			AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
			Count:           int64(tier.Count),
			Selectors: []resourcev1.DeviceSelector{{
				CEL: &resourcev1.CELDeviceSelector{Expression: r.tierBounds(tier)},
			}},
		}
		if r.WholeNodeClaims {
			subrequests[i].AllocationMode = resourcev1.DeviceAllocationModeAll
			subrequests[i].Count = 0
		}
	}

	return &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      claimName,
			Namespace: r.Namespace,
			// No role, owner, or shape labels: seating is the ledger's record,
			// and the allocated shape is DRA's answer in status.
			Labels: map[string]string{LabelManaged: "true"},
		},
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{{
					Name:           podClaimName,
					FirstAvailable: subrequests,
				}},
			},
		},
	}
}

// tierBounds is the CEL that admits exactly the devices a tier was priced
// on: a capacity comparison when the driver publishes memory, else membership
// in the memory table's values for that size bucket.
func (r *WorkloadReconciler) tierBounds(tier placement.Tier) string {
	if !r.DeviceMemoryTable.Enabled() {
		return fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("%d")) >= 0 && device.capacity["%s"].memory.compareTo(quantity("%dGi")) <= 0`,
			r.DeviceDriver, tier.FloorBytes, r.DeviceDriver, placement.CeilGiB(tier.CeilingBytes))
	}
	values := r.DeviceMemoryTable.ValuesWithin(tier.FloorBytes, tier.CeilingBytes)
	quoted := make([]string, len(values))
	for i, value := range values {
		quoted[i] = fmt.Sprintf("%q", value)
	}
	return fmt.Sprintf(`device.attributes["%s"].%s in [%s]`, r.DeviceDriver, r.DeviceMemoryTable.Attribute, strings.Join(quoted, ", "))
}

// workerContainerName is the container in the template that consumes the
// accelerator: where the claim reference and time-slice group land.
func workerContainerName(worker *openrlv1alpha1.Workload) string {
	if worker.Spec.WorkerContainerName != "" {
		return worker.Spec.WorkerContainerName
	}
	return "worker"
}

// validateTemplate rejects placement-owned fields in the inline template:
// they are placement's to decide, so they are refused rather than merged.
// (CRD CEL catches the pod-level ones at admission; this is the backstop.)
func validateTemplate(worker *openrlv1alpha1.Workload) error {
	spec := &worker.Spec.Template.Spec
	switch {
	case len(spec.Containers) == 0:
		return fmt.Errorf("template declares no containers")
	case spec.NodeName != "":
		return fmt.Errorf("template sets nodeName; node choice is placement's")
	case spec.NodeSelector != nil:
		return fmt.Errorf("template sets nodeSelector; node choice is placement's")
	case spec.Affinity != nil:
		return fmt.Errorf("template sets affinity; node choice is placement's")
	case len(spec.ResourceClaims) > 0:
		return fmt.Errorf("template references resource claims; the claim is placement's")
	}

	found := false
	workerName := workerContainerName(worker)
	for i := range spec.Containers {
		c := &spec.Containers[i]
		if c.Name == workerName {
			found = true
		}
		if len(c.Resources.Claims) > 0 {
			return fmt.Errorf("container %s references resource claims; the claim is placement's", c.Name)
		}
		for name := range c.Resources.Requests {
			if strings.Contains(string(name), "/") {
				return fmt.Errorf("container %s requests extended resource %s; accelerators come from the claim", c.Name, name)
			}
		}
		for name := range c.Resources.Limits {
			if strings.Contains(string(name), "/") {
				return fmt.Errorf("container %s limits extended resource %s; accelerators come from the claim", c.Name, name)
			}
		}
		for _, env := range c.Env {
			if env.Name == timeSliceGroupEnv {
				return fmt.Errorf("container %s sets %s; the ledger is the claim name and is stamped by the controller", c.Name, timeSliceGroupEnv)
			}
		}
	}
	if !found {
		return fmt.Errorf("template has no container named %q (spec.workerContainerName)", workerName)
	}
	return nil
}

// renderPod builds the worker pod: the spec's inline template for everything
// knowable before placement, the controller's stamps for everything
// placement decided. The template was already validated at the top of
// place(), the only path here.
func (r *WorkloadReconciler) renderPod(worker *openrlv1alpha1.Workload, podName, claimName string, prefs []placement.NodePreference) (*corev1.Pod, error) {
	template := worker.Spec.Template.DeepCopy()
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        podName,
			Namespace:   r.Namespace,
			Labels:      template.Labels,
			Annotations: template.Annotations,
		},
		Spec: template.Spec,
	}
	attachClaim(pod, worker, claimName)
	attachTimeSliceGroup(pod, worker, claimName)

	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels["app"] = "open-rl-" + string(worker.Spec.Role) + "-worker"
	pod.Labels[LabelClaim] = claimName
	// Label values cap at 63 characters and forbid dots; names do neither.
	// Labels carry sanitized identities, env vars carry the full ones.
	pod.Labels[LabelWorker] = sanitizeLabel(worker.Name)
	pod.Labels[LabelRole] = string(worker.Spec.Role)

	// Constraining the pod is what constrains where its claim can land.
	// Two ORed terms, because a node selector cannot say "or unlabeled":
	// the documented default is that a node naming no role labels takes both.
	pod.Spec.Affinity = &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
			NodeSelectorTerms: []corev1.NodeSelectorTerm{
				{MatchExpressions: []corev1.NodeSelectorRequirement{
					{Key: NodeLabelEnabled, Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
					{Key: nodeRoleLabel[worker.Spec.Role], Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
				}},
				{MatchExpressions: []corev1.NodeSelectorRequirement{
					{Key: NodeLabelEnabled, Operator: corev1.NodeSelectorOpIn, Values: []string{"true"}},
					{Key: NodeLabelTrainer, Operator: corev1.NodeSelectorOpDoesNotExist},
					{Key: NodeLabelSampler, Operator: corev1.NodeSelectorOpDoesNotExist},
				}},
			},
		},
		// The tiers order device sizes inside the claim, but firstAvailable
		// only ranks alternatives within one node's inventory; the node
		// choice itself is kube-scheduler's, and its scoring is size-blind.
		// These weights carry the same order across nodes. Preferences never
		// gate: a full or vanished node just scores zero. matchFields
		// requires exactly one value, hence one term per node.
		PreferredDuringSchedulingIgnoredDuringExecution: preferredTerms(prefs),
	}}

	if err := controllerutil.SetControllerReference(worker, pod, r.Scheme()); err != nil {
		return nil, fmt.Errorf("set owner of pod %s: %w", podName, err)
	}
	return pod, nil
}

// preferredTerms renders the tight-fit rungs as weighted scheduling terms.
func preferredTerms(prefs []placement.NodePreference) []corev1.PreferredSchedulingTerm {
	var terms []corev1.PreferredSchedulingTerm
	for _, pref := range prefs {
		for _, name := range pref.Nodes {
			terms = append(terms, corev1.PreferredSchedulingTerm{
				Weight: pref.Weight,
				Preference: corev1.NodeSelectorTerm{
					MatchFields: []corev1.NodeSelectorRequirement{{
						Key: "metadata.name", Operator: corev1.NodeSelectorOpIn, Values: []string{name},
					}},
				},
			})
		}
	}
	return terms
}

// setEnv merges one variable by name, overwriting whatever the template had.
func setEnv(container *corev1.Container, want corev1.EnvVar) {
	for i := range container.Env {
		if container.Env[i].Name == want.Name {
			container.Env[i] = want
			return
		}
	}
	container.Env = append(container.Env, want)
}

// attachClaim points the pod at its ResourceClaim: the pod-level entry, and
// the consumption reference on the worker container -- the one the spec
// named as the accelerator consumer.
func attachClaim(pod *corev1.Pod, worker *openrlv1alpha1.Workload, claimName string) {
	pod.Spec.ResourceClaims = []corev1.PodResourceClaim{{
		Name:              podClaimName,
		ResourceClaimName: &claimName,
	}}
	name := workerContainerName(worker)
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == name {
			pod.Spec.Containers[i].Resources.Claims = []corev1.ResourceClaim{{Name: podClaimName}}
		}
	}
}

// attachTimeSliceGroup stamps what only placement knows: the time-slice
// ledger (the claim name) and the booking identity. Owner/workload-id env
// rides in the template the API server authored.
func attachTimeSliceGroup(pod *corev1.Pod, worker *openrlv1alpha1.Workload, claimName string) {
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	request := requestFrom(worker)

	pod.Labels[timeSliceEnabledLabel] = "true"
	pod.Labels[timeSliceGroupLabel] = claimName
	pod.Labels[timeSliceOwnerLabel] = sanitizeLabel(request.OwnerKey())
	pod.Labels[timeSliceJobIDLabel] = sanitizeLabel(request.WorkerID)
	name := workerContainerName(worker)
	for i := range pod.Spec.Containers {
		if pod.Spec.Containers[i].Name == name {
			setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: timeSliceGroupEnv, Value: claimName})
			// The booking's identity, so the runtime can refuse a pod whose
			// assignment has been superseded. place() guarantees a live seat
			// before any pod is created, so these are always set.
			setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: claimLedgerEnv, Value: ledgerNameFor(claimName)})
			setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: assignmentIDEnv, Value: worker.Status.AssignmentID})
		}
	}
}

// unschedulableSince is when kube-scheduler first refused the pod, or zero.
func unschedulableSince(pod *corev1.Pod) time.Time {
	if pod.Status.Phase != corev1.PodPending && pod.Status.Phase != "" {
		return time.Time{}
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodScheduled && condition.Status == corev1.ConditionFalse {
			return condition.LastTransitionTime.Time
		}
	}
	return time.Time{}
}

// unschedulableMessage is the scheduler's reason a pod cannot be placed, if it
// gave one.
func unschedulableMessage(pod *corev1.Pod) string {
	if pod.Status.Phase != corev1.PodPending && pod.Status.Phase != "" {
		return ""
	}
	for _, condition := range pod.Status.Conditions {
		if condition.Type != corev1.PodScheduled || condition.Status != corev1.ConditionFalse {
			continue
		}
		detail := condition.Message
		if detail == "" {
			detail = condition.Reason
		}
		if detail != "" {
			return "Unschedulable: " + detail
		}
	}
	return ""
}
