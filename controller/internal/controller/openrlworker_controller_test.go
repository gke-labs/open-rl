package controller

import (
	"context"
	"strconv"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
)

const (
	testNamespace = "open-rl"
	testDriver    = "gpu.nvidia.com"
	testNode      = "node-a"
	testTemplate  = "trainer-pod-template"
)

// podTemplateYAML is deliberately hostile to the controller: it pins a
// ResourceClaim and a nodeSelector of its own. Both must be overwritten, since
// picking hardware is the whole job of this controller.
const podTemplateYAML = `
apiVersion: v1
kind: Pod
spec:
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-l4
  resourceClaims:
  - name: gpu
    resourceClaimName: someone-elses-claim
  containers:
  - name: worker
    image: template-image
    env:
    - name: KEEP_ME
      value: "1"
`

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add client-go scheme: %v", err)
	}
	if err := openrlv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add openrl scheme: %v", err)
	}
	return scheme
}

// enabledNode is a pool the operator has opted in for both roles, described by
// a ResourceSlice with two 96Gi devices. maxResidents is how many time-sliced
// workers the operator will let share one claim here; 1 means no sharing, which
// is also what an unlabelled node gets.
func enabledNode(maxResidents int) []client.Object {
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNode,
			Labels: map[string]string{
				NodeLabelEnabled:      "true",
				NodeLabelTrainer:      "true",
				NodeLabelSampler:      "true",
				NodeLabelMaxResidents: strconv.Itoa(maxResidents),
			},
		},
		Status: corev1.NodeStatus{
			Allocatable: corev1.ResourceList{corev1.ResourceMemory: resource.MustParse("340Gi")},
		},
	}

	nodeName := testNode
	product := "NVIDIA RTX PRO 6000 Blackwell"
	device := func(name string) resourcev1.Device {
		return resourcev1.Device{
			Name: name,
			Attributes: map[resourcev1.QualifiedName]resourcev1.DeviceAttribute{
				"productName": {StringValue: &product},
			},
			Capacity: map[resourcev1.QualifiedName]resourcev1.DeviceCapacity{
				"memory": {Value: resource.MustParse("96Gi")},
			},
		}
	}
	slice := &resourcev1.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: "slice-a"},
		Spec: resourcev1.ResourceSliceSpec{
			Driver:   testDriver,
			NodeName: &nodeName,
			Pool:     resourcev1.ResourcePool{Name: testNode, ResourceSliceCount: 1},
			Devices:  []resourcev1.Device{device("gpu-0"), device("gpu-1")},
		},
	}

	// The key is deliberately not the controller's "pod.yaml" default: the
	// deployed templates are shared with the static worker manager and keep
	// its key names, so every test also exercises the lone-key fallback.
	return []client.Object{node, slice, &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: testTemplate, Namespace: testNamespace},
		Data:       map[string]string{"trainer-worker-pod.yaml": podTemplateYAML},
	}}
}

// worker is the whole of a request: a role, an id, how much accelerator memory
// it needs, and -- if it shares weights with anyone -- the cohort it shares
// them with. Everything else the scheduler derives.
func worker(name, modelID string, role openrlv1alpha1.WorkerRole, memory string) *openrlv1alpha1.OpenRLWorker {
	return &openrlv1alpha1.OpenRLWorker{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         testNamespace,
			CreationTimestamp: metav1.Now(),
		},
		Spec: openrlv1alpha1.OpenRLWorkerSpec{
			Role:    role,
			ModelID: modelID,
			Memory:  resource.MustParse(memory),
		},
	}
}

// trainerWorker shares nothing, so it is a cohort of one: it takes its turn
// alone against every other resident of its claim.
func trainerWorker(name, modelID string) *openrlv1alpha1.OpenRLWorker {
	return worker(name, modelID, openrlv1alpha1.RoleTrainer, "24Gi")
}

// loraWorker names the base model it sits on. Adapters over one frozen copy
// name the same cohort and are resident together, their memory summing.
func loraWorker(name, modelID, cohort string) *openrlv1alpha1.OpenRLWorker {
	w := worker(name, modelID, openrlv1alpha1.RoleTrainer, "24Gi")
	w.Spec.Cohort = cohort
	return w
}

func newReconciler(t *testing.T, objects ...client.Object) *OpenRLWorkerReconciler {
	t.Helper()
	c := fake.NewClientBuilder().
		WithScheme(testScheme(t)).
		WithObjects(objects...).
		WithStatusSubresource(&openrlv1alpha1.OpenRLWorker{}).
		Build()
	return &OpenRLWorkerReconciler{
		Client:       c,
		Namespace:    testNamespace,
		DeviceClass:  testDriver,
		DeviceDriver: testDriver,
		DefaultPodTemplates: map[openrlv1alpha1.WorkerRole]string{
			openrlv1alpha1.RoleTrainer: testTemplate,
			openrlv1alpha1.RoleSampler: testTemplate,
		},
		RetryInterval:    time.Second,
		PlacementTimeout: time.Hour,
		ReclaimInterval:  time.Minute,
	}
}

func runReconcile(t *testing.T, r *OpenRLWorkerReconciler, name string) ctrl.Result {
	t.Helper()
	result, err := r.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: types.NamespacedName{Namespace: testNamespace, Name: name},
	})
	if err != nil {
		t.Fatalf("reconcile %s: %v", name, err)
	}
	return result
}

// settle places the worker and creates its pod: the claim is cut on one pass
// and the pod built on the next.
func settle(t *testing.T, r *OpenRLWorkerReconciler, names ...string) {
	t.Helper()
	for _, name := range names {
		runReconcile(t, r, name)
		runReconcile(t, r, name)
	}
}

func getWorker(t *testing.T, r *OpenRLWorkerReconciler, name string) *openrlv1alpha1.OpenRLWorker {
	t.Helper()
	var w openrlv1alpha1.OpenRLWorker
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &w); err != nil {
		t.Fatalf("get worker %s: %v", name, err)
	}
	return &w
}

func claimOf(t *testing.T, r *OpenRLWorkerReconciler, name string) string {
	t.Helper()
	claim := getWorker(t, r, name).Status.ClaimName
	if claim == "" {
		t.Fatalf("worker %s was not placed", name)
	}
	return claim
}

func getPod(t *testing.T, r *OpenRLWorkerReconciler, name string) *corev1.Pod {
	t.Helper()
	var pod corev1.Pod
	err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &pod)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		t.Fatalf("get pod %s: %v", name, err)
	}
	return &pod
}

func envOf(container corev1.Container, name string) string {
	for _, env := range container.Env {
		if env.Name == name {
			return env.Value
		}
	}
	return ""
}

// A worker with nothing placed yet gets a claim and a pod, and the pod is
// rendered against the claim rather than against whatever the template said.
func TestReconcilePlacesUnplacedWorker(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

	runReconcile(t, r, "w-a")

	placed := getWorker(t, r, "w-a")
	if placed.Status.Phase != openrlv1alpha1.PhasePlacing {
		t.Fatalf("phase = %q, want Placing", placed.Status.Phase)
	}
	claimName := placed.Status.ClaimName
	if claimName == "" {
		t.Fatal("no claim recorded on status")
	}

	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: claimName}, &claim); err != nil {
		t.Fatalf("claim %s was not created: %v", claimName, err)
	}
	if claim.Labels[LabelManaged] != "true" {
		t.Errorf("claim is missing the managed label: %v", claim.Labels)
	}

	// The pod is created on the pass after the claim, so reconcile again.
	runReconcile(t, r, "w-a")
	pod := getPod(t, r, "open-rl-trainer-model-a")
	if pod == nil {
		t.Fatal("pod was not created")
	}
	if got := pod.Labels[LabelClaim]; got != claimName {
		t.Errorf("pod claim label = %q, want %q", got, claimName)
	}
	if len(pod.Spec.ResourceClaims) != 1 || *pod.Spec.ResourceClaims[0].ResourceClaimName != claimName {
		t.Errorf("pod resourceClaims = %+v, want the template's claim replaced by %q", pod.Spec.ResourceClaims, claimName)
	}
	if _, pinned := pod.Spec.NodeSelector["cloud.google.com/gke-accelerator"]; pinned {
		t.Errorf("the template's accelerator nodeSelector survived: %v", pod.Spec.NodeSelector)
	}
	if pod.Spec.NodeSelector[NodeLabelTrainer] != "true" || pod.Spec.NodeSelector[NodeLabelEnabled] != "true" {
		t.Errorf("nodeSelector = %v, want the role and enabled labels", pod.Spec.NodeSelector)
	}

	// The group is the claim -- not a cluster-wide "trainers" bucket -- and a
	// worker that named no cohort is a cohort of one: it takes a turn against
	// every other resident of the claim.
	if pod.Labels[timeSliceEnabledLabel] != "true" || pod.Labels[timeSliceGroupLabel] != claimName {
		t.Errorf("time-slice labels = %v, want enabled with group %q", pod.Labels, claimName)
	}
	if got := pod.Labels[timeSliceCohortLabel]; got != "model-a" {
		t.Errorf("cohort label = %q, want the worker's own id", got)
	}
	container := pod.Spec.Containers[0]
	if got := envOf(container, timeSliceGroupEnv); got != claimName {
		t.Errorf("%s = %q, want %q", timeSliceGroupEnv, got, claimName)
	}
	if got := envOf(container, timeSliceCohortEnv); got != "model-a" {
		t.Errorf("%s = %q, want %q", timeSliceCohortEnv, got, "model-a")
	}
	if got := envOf(container, "KEEP_ME"); got != "1" {
		t.Errorf("the template's own env was dropped: %v", container.Env)
	}
}

// The device count is derived from memory, never asked for. There is no
// sharding: the model is laid out layer by layer over whatever it is given, so
// 120Gi on 96Gi devices needs two of them and half the model sits on each.
func TestReconcileDerivesTheDeviceCountFromMemory(t *testing.T) {
	big := worker("w-big", "model-big", openrlv1alpha1.RoleTrainer, "120Gi")
	r := newReconciler(t, append(enabledNode(4), big)...)

	runReconcile(t, r, "w-big")

	status := getWorker(t, r, "w-big").Status
	if status.DeviceCount != 2 {
		t.Errorf("deviceCount = %d, want 2: 120Gi does not fit one 96Gi device", status.DeviceCount)
	}
	if status.MemoryPerDevice != "60Gi" {
		t.Errorf("memoryPerDevice = %q, want 60Gi", status.MemoryPerDevice)
	}
	// Parking moves the whole footprint to host RAM, however it was spread.
	if status.HostMemoryPerResident != "120Gi" {
		t.Errorf("hostMemoryPerResident = %q, want 120Gi", status.HostMemoryPerResident)
	}
}

// The regression this test exists for: spec.resourceClaims is immutable, so a
// worker re-placed onto a different claim can never be reached by its existing
// pod. The controller has to delete it, and the next pass has to build one
// against the new claim.
func TestReconcileRecreatesPodBoundToAStaleClaim(t *testing.T) {
	w := trainerWorker("w-a", "model-a")
	w.Status = openrlv1alpha1.OpenRLWorkerStatus{
		Phase:     openrlv1alpha1.PhaseRunning,
		ClaimName: "claim-gone",
		PodName:   "open-rl-trainer-model-a",
	}
	stale := "claim-gone"
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "open-rl-trainer-model-a",
			Namespace: testNamespace,
			Labels:    map[string]string{LabelClaim: stale, LabelWorker: "w-a"},
		},
		Spec: corev1.PodSpec{
			Containers:     []corev1.Container{{Name: "worker", Image: "template-image"}},
			ResourceClaims: []corev1.PodResourceClaim{{Name: podClaimName, ResourceClaimName: &stale}},
		},
	}
	r := newReconciler(t, append(enabledNode(4), w, pod)...)

	runReconcile(t, r, "w-a")

	if getPod(t, r, "open-rl-trainer-model-a") != nil {
		t.Fatal("the pod bound to the vanished claim was left in place")
	}
	after := getWorker(t, r, "w-a")
	if after.Status.Reason != "RecreatingPodOnNewClaim" {
		t.Errorf("reason = %q, want RecreatingPodOnNewClaim", after.Status.Reason)
	}
	if after.Status.ClaimName == stale || after.Status.ClaimName == "" {
		t.Fatalf("claim = %q, want a freshly cut one", after.Status.ClaimName)
	}

	// Converge: the next pass builds a pod against the claim the worker now holds.
	runReconcile(t, r, "w-a")
	rebuilt := getPod(t, r, "open-rl-trainer-model-a")
	if rebuilt == nil {
		t.Fatal("no pod was rebuilt")
	}
	if got := *rebuilt.Spec.ResourceClaims[0].ResourceClaimName; got != after.Status.ClaimName {
		t.Errorf("rebuilt pod names claim %q, want %q", got, after.Status.ClaimName)
	}
}

// A pod that is already on the right claim is left alone. Without this, the
// delete branch above would restart every worker on every reconcile.
func TestReconcileLeavesAMatchingPodAlone(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	created := getPod(t, r, "open-rl-trainer-model-a")
	if created == nil {
		t.Fatal("pod was not created")
	}

	for i := 0; i < 3; i++ {
		runReconcile(t, r, "w-a")
	}
	still := getPod(t, r, "open-rl-trainer-model-a")
	if still == nil {
		t.Fatal("a settled pod was deleted")
	}
	if still.UID != created.UID {
		t.Errorf("pod was recreated (uid %q -> %q)", created.UID, still.UID)
	}
}

// Two workers that share nothing still share a claim: a claim is a bundle of
// accelerators, and two cohorts on one bundle take turns rather than being
// pushed onto hardware of their own.
func TestReconcileSharesAClaimBetweenTwoSoloTrainers(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	runReconcile(t, r, "w-a")
	runReconcile(t, r, "w-b")

	if a, b := claimOf(t, r, "w-a"), claimOf(t, r, "w-b"); a != b {
		t.Errorf("claims %q and %q differ; the second worker should have joined the first's claim", a, b)
	}

	var claims resourcev1.ResourceClaimList
	if err := r.List(context.Background(), &claims, client.InNamespace(testNamespace)); err != nil {
		t.Fatalf("list claims: %v", err)
	}
	if len(claims.Items) != 1 {
		t.Errorf("cut %d claims, want 1", len(claims.Items))
	}
}

// A trainer and a sampler on one accelerator. Role selects which node pools may
// host a worker and nothing else -- in particular it does not partition claims
// -- so on a node labelled for both, the two halves of the loop take turns on
// one GPU instead of demanding one each.
func TestReconcileRunsATrainerAndASamplerOnOneClaim(t *testing.T) {
	trainer := trainerWorker("w-t", "model-t")
	sampler := worker("w-s", "model-s", openrlv1alpha1.RoleSampler, "24Gi")
	r := newReconciler(t, append(enabledNode(4), trainer, sampler)...)

	settle(t, r, "w-t", "w-s")

	claim := claimOf(t, r, "w-t")
	if got := claimOf(t, r, "w-s"); got != claim {
		t.Fatalf("sampler landed on %q and trainer on %q, want one claim", got, claim)
	}

	// Same group, different cohorts: they share the bundle, not the weights.
	trainerPod, samplerPod := getPod(t, r, "open-rl-trainer-model-t"), getPod(t, r, "open-rl-sampler-model-s")
	if trainerPod == nil || samplerPod == nil {
		t.Fatalf("pods missing: trainer=%v sampler=%v", trainerPod != nil, samplerPod != nil)
	}
	if trainerPod.Labels[timeSliceGroupLabel] != samplerPod.Labels[timeSliceGroupLabel] {
		t.Errorf("time-slice groups differ: %q and %q", trainerPod.Labels[timeSliceGroupLabel], samplerPod.Labels[timeSliceGroupLabel])
	}
	if trainerPod.Labels[timeSliceCohortLabel] == samplerPod.Labels[timeSliceCohortLabel] {
		t.Errorf("both pods are in cohort %q, but they share nothing", trainerPod.Labels[timeSliceCohortLabel])
	}
	// The sampler still may not land on a pool the operator closed to samplers.
	if samplerPod.Spec.NodeSelector[NodeLabelSampler] != "true" {
		t.Errorf("sampler nodeSelector = %v, want the sampler role label", samplerPod.Spec.NodeSelector)
	}
}

// Three adapters on one claim, two base models between them. The cohort is a
// string the caller chooses and the scheduler only ever compares: same string
// means resident together, different string means take turns. The controller
// has no table of which workload kinds may share.
func TestReconcileGroupsWorkersByTheirCohortString(t *testing.T) {
	workers := []client.Object{
		loraWorker("w-a", "model-a", "Qwen/Qwen3-0.6B"),
		loraWorker("w-b", "model-b", "Qwen/Qwen3-0.6B"),
		loraWorker("w-c", "model-c", "meta-llama/Llama-3-8B"),
	}
	r := newReconciler(t, append(enabledNode(4), workers...)...)

	settle(t, r, "w-a", "w-b", "w-c")

	claim := claimOf(t, r, "w-a")
	if b, c := claimOf(t, r, "w-b"), claimOf(t, r, "w-c"); b != claim || c != claim {
		t.Fatalf("workers landed on claims %q, %q and %q, want one claim for all three", claim, b, c)
	}

	cohort := func(model string) string {
		pod := getPod(t, r, "open-rl-trainer-"+model)
		if pod == nil {
			t.Fatalf("no pod for %s", model)
		}
		return pod.Labels[timeSliceCohortLabel]
	}
	if a, b := cohort("model-a"), cohort("model-b"); a != b {
		t.Errorf("cohorts %q and %q differ, but both adapters named the same base model", a, b)
	}
	if a, c := cohort("model-a"), cohort("model-c"); a == c {
		t.Errorf("both pods are in cohort %q, but they share no base weights to be resident together", a)
	}
}

// The same two trainers on a pool that seats one resident get a claim each.
// openrl.io/max-residents is what turns sharing on, and a node without the
// label defaults to no sharing rather than to unlimited.
func TestReconcileDoesNotShareWhenTheNodeSeatsOneResident(t *testing.T) {
	r := newReconciler(t, append(enabledNode(1), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	runReconcile(t, r, "w-a")
	runReconcile(t, r, "w-b")

	if a, b := claimOf(t, r, "w-a"), claimOf(t, r, "w-b"); a == b {
		t.Errorf("both workers landed on claim %q, but the pool seats one resident", a)
	}
}

// A worker asking for more than any registered pool can provide is reported as
// pending with an explanation, not silently dropped or placed anyway.
func TestReconcileReportsAnUnplaceableWorkerAsPending(t *testing.T) {
	r := newReconciler(t, append(enabledNode(4), worker("w-a", "model-a", openrlv1alpha1.RoleTrainer, "4000Gi"))...)

	result := runReconcile(t, r, "w-a")
	if result.RequeueAfter != r.RetryInterval {
		t.Errorf("requeueAfter = %v, want %v", result.RequeueAfter, r.RetryInterval)
	}
	after := getWorker(t, r, "w-a")
	if after.Status.Phase != openrlv1alpha1.PhasePending {
		t.Fatalf("phase = %q, want Pending", after.Status.Phase)
	}
	if after.Status.Reason == "" {
		t.Error("a pending worker should carry an explanation")
	}
	if getPod(t, r, "open-rl-trainer-model-a") != nil {
		t.Error("a pod was created for a worker that was never placed")
	}
}
