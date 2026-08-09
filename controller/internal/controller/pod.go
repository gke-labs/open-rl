package controller

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/yaml"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// Time-slicer contract, mirrored from src/accel_timeslicer/workload.py. A pod
// label so other pods can discover the workload, an env var so the process
// itself knows which group and cohort it belongs to. All of them must agree.
const (
	timeSliceEnabledLabel = "accel-timeslicer"
	timeSliceGroupLabel   = "timeslice.io/group"
	timeSliceCohortLabel  = "timeslice.io/cohort"
	timeSliceJobIDLabel   = "timeslice.io/job-id"
	timeSliceJobIDEnv     = "OPEN_RL_TIME_SLICE_JOB_ID"
	timeSliceGroupEnv     = "OPEN_RL_TIME_SLICE_GROUP"
	timeSliceCohortEnv    = "OPEN_RL_TIME_SLICE_COHORT"
)

// The name the pod and the claim agree to call the allocation.
const podClaimName = "gpu"

var podNamePrefix = map[openrlv1alpha1.WorkerRole]string{
	openrlv1alpha1.RoleTrainer: "open-rl-trainer-",
	openrlv1alpha1.RoleSampler: "open-rl-sampler-",
}

// labelUnsafe matches everything a DNS-1123 pod name may not contain. Label
// values are laxer, but the sanitized id is reused in the pod name, so the
// stricter rule governs both.
var labelUnsafe = regexp.MustCompile(`[^a-z0-9-]+`)

// sanitizeLabel reduces an arbitrary id to something usable as a label value
// and as part of a pod name. Empty in, empty out.
func sanitizeLabel(value string) string {
	cleaned := strings.Trim(labelUnsafe.ReplaceAllString(strings.ToLower(value), "-"), "-")
	if len(cleaned) > 63 {
		cleaned = strings.TrimRight(cleaned[:63], "-")
	}
	return cleaned
}

// workerPodName is the one name a worker pod gets, whichever manager creates
// it. Matching src/server/k8s_worker_manager.py keeps the gateway's static
// path and this controller from ever creating two pods for one model.
func workerPodName(modelID string, role openrlv1alpha1.WorkerRole) (string, error) {
	id := sanitizeLabel(modelID)
	if id == "" {
		return "", fmt.Errorf("modelId %q has no label-safe characters", modelID)
	}
	prefix, ok := podNamePrefix[role]
	if !ok {
		return "", fmt.Errorf("unknown role %q", role)
	}
	name := prefix + id
	if len(name) > 253 {
		name = name[:253]
	}
	return name, nil
}

// claimNameFor is the claim a worker cuts when it cannot join one. Derived
// from the worker so a repeated reconcile converges on the same name rather
// than cutting a second claim.
func claimNameFor(worker *openrlv1alpha1.OpenRLWorker) string {
	name := "claim-" + worker.Name
	if len(name) > 253 {
		name = name[:253]
	}
	return name
}

// buildClaim builds a ResourceClaim for a shape no existing claim satisfies.
//
// The controller picks the device count; the per-device floor goes into a CEL
// selector and DRA finds devices that clear it. Deliberately not a node
// selector: which node satisfies this is the scheduler's decision, informed by
// the pod's own nodeSelector.
func (r *OpenRLWorkerReconciler) buildClaim(worker *openrlv1alpha1.OpenRLWorker, claim *placement.Claim, perDeviceBytes int64) *resourcev1.ResourceClaim {
	// No role, kind or cohort label: a claim is just a bundle of accelerators,
	// and which cohorts sit on it is rebuilt from the workers that reference it.
	labels := map[string]string{
		LabelManaged:    "true",
		LabelAccelCount: fmt.Sprint(claim.DeviceCount),
	}

	floor := fmt.Sprintf(`device.capacity["%s"].memory.compareTo(quantity("%dGi")) >= 0`, r.DeviceDriver, placement.CeilGiB(perDeviceBytes))

	return &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      claim.Name,
			Namespace: r.Namespace,
			Labels:    labels,
		},
		Spec: resourcev1.ResourceClaimSpec{
			Devices: resourcev1.DeviceClaim{
				Requests: []resourcev1.DeviceRequest{{
					Name: podClaimName,
					Exactly: &resourcev1.ExactDeviceRequest{
						DeviceClassName: r.DeviceClass,
						Count:           int64(claim.DeviceCount),
						AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
						Selectors: []resourcev1.DeviceSelector{{
							CEL: &resourcev1.CELDeviceSelector{Expression: floor},
						}},
					},
				}},
			},
		},
	}
}

// renderPod builds the worker pod: the operator's template for everything
// placement has no opinion about, the controller's decision for everything it
// does.
func (r *OpenRLWorkerReconciler) renderPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, podName, claimName string) (*corev1.Pod, error) {
	pod, err := r.loadTemplate(ctx, worker)
	if err != nil {
		return nil, err
	}
	if len(pod.Spec.Containers) == 0 {
		return nil, fmt.Errorf("pod template for role %s declares no containers", worker.Spec.Role)
	}

	pod.Name = podName
	pod.Namespace = r.Namespace
	applyOverlay(&pod.Spec.Containers[0], worker.Spec.Container)
	attachClaim(pod, claimName)
	attachTimeSliceGroup(pod, worker, claimName)

	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels["app"] = "open-rl-" + string(worker.Spec.Role) + "-worker"
	pod.Labels[LabelClaim] = claimName
	pod.Labels[LabelWorker] = worker.Name
	pod.Labels[LabelRole] = string(worker.Spec.Role)

	// The role lives on the pod, not on the claim: DRA has no notion of node
	// labels, so constraining the pod is what constrains where its claim can
	// land. Whatever accelerator SKU the template pinned is dropped -- picking
	// hardware is this controller's job now.
	pod.Spec.NodeSelector = map[string]string{
		NodeLabelEnabled:                "true",
		nodeRoleLabel[worker.Spec.Role]: "true",
	}

	if err := controllerutil.SetControllerReference(worker, pod, r.Scheme()); err != nil {
		return nil, fmt.Errorf("set owner of pod %s: %w", podName, err)
	}
	return pod, nil
}

// loadTemplate reads the pod YAML the worker asked for, or the controller's
// role default.
func (r *OpenRLWorkerReconciler) loadTemplate(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker) (*corev1.Pod, error) {
	name, key := r.DefaultPodTemplates[worker.Spec.Role], "pod.yaml"
	if ref := worker.Spec.PodTemplate; ref != nil {
		name = ref.Name
		if ref.Key != "" {
			key = ref.Key
		}
	}
	if name == "" {
		return nil, fmt.Errorf("no pod template configured for role %s and none given in spec.podTemplate", worker.Spec.Role)
	}

	var cm corev1.ConfigMap
	if err := r.Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: name}, &cm); err != nil {
		return nil, fmt.Errorf("read pod template ConfigMap %s: %w", name, err)
	}
	raw, ok := cm.Data[key]
	if !ok && len(cm.Data) == 1 {
		// The default key is a convention, not a contract: the deployed
		// templates are shared with the static worker manager and keep its key
		// names. A single-key ConfigMap is unambiguous, so take the lone entry.
		for _, raw = range cm.Data {
			ok = true
		}
	}
	if !ok {
		return nil, fmt.Errorf("pod template ConfigMap %s has no key %q and does not have exactly one key", name, key)
	}

	var pod corev1.Pod
	if err := yaml.Unmarshal([]byte(raw), &pod); err != nil {
		return nil, fmt.Errorf("parse pod template %s/%s: %w", name, key, err)
	}
	return &pod, nil
}

// applyOverlay stamps the gateway's per-model decisions onto the container.
//
// Everything the pod needs to know about the model arrives on the CRD, so the
// controller never has to reach into the gateway's metadata store to render a
// pod.
func applyOverlay(container *corev1.Container, overlay *openrlv1alpha1.ContainerOverlay) {
	if overlay == nil {
		return
	}
	if overlay.Image != "" {
		container.Image = overlay.Image
	}
	if len(overlay.Command) > 0 {
		container.Command = overlay.Command
	}
	container.Args = append(container.Args, overlay.Args...)
	for _, env := range overlay.Env {
		setEnv(container, env)
	}
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

// attachClaim points the pod at a specific ResourceClaim.
//
// Replaces whatever claim the template carried rather than adding to it: a pod
// holding two GPU claims is pinned to the intersection of two allocations.
func attachClaim(pod *corev1.Pod, claimName string) {
	pod.Spec.ResourceClaims = []corev1.PodResourceClaim{{
		Name:              podClaimName,
		ResourceClaimName: &claimName,
	}}
	for i := range pod.Spec.Containers {
		pod.Spec.Containers[i].Resources.Claims = []corev1.ResourceClaim{{Name: podClaimName}}
	}
}

// attachTimeSliceGroup tells the node-local time-slicer which accelerator
// bundle this worker shares, and who it shares it *with*.
//
// The claim is the bundle, so the claim is the group. This replaces the fixed
// per-role group name the static path used ("trainers", "samplers"), which put
// every trainer in the cluster into one group and made workers on unrelated
// allocations take turns against each other for no reason.
//
// The cohort is the finer half: turns are taken between cohorts, not between
// processes. Every kind opts in, because even a kind whose residents run
// concurrently only does so within one cohort -- two lora workers on different
// base models still have to swap.
func attachTimeSliceGroup(pod *corev1.Pod, worker *openrlv1alpha1.OpenRLWorker, claimName string) {
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	jobID := string(worker.Spec.Role) + "-" + sanitizeLabel(worker.Spec.ModelID)
	cohort := requestFrom(worker).CohortKey()

	pod.Labels[timeSliceEnabledLabel] = "true"
	pod.Labels[timeSliceGroupLabel] = claimName
	pod.Labels[timeSliceCohortLabel] = cohort
	pod.Labels[timeSliceJobIDLabel] = jobID
	for i := range pod.Spec.Containers {
		setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: timeSliceGroupEnv, Value: claimName})
		setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: timeSliceCohortEnv, Value: cohort})
		setEnv(&pod.Spec.Containers[i], corev1.EnvVar{Name: timeSliceJobIDEnv, Value: jobID})
	}
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
