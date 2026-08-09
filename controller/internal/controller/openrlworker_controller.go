package controller

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/controller/internal/placement"
)

// SwitchCostWarnThreshold is the point past which a shared claim is worth
// complaining about. A worker this expensive to park costs the whole group a
// visible share of every slice.
const SwitchCostWarnThreshold = 15 * time.Second

// OpenRLWorkerReconciler turns OpenRLWorker requests into ResourceClaims and
// pods. The decision lives in internal/placement; this is the part that reads
// and writes Kubernetes objects.
//
// It runs with a concurrency of one. Placement is a global bin-packing
// decision over the whole fleet, so two reconciles running at once would each
// decide against a fleet that does not include the other's booking.
type OpenRLWorkerReconciler struct {
	client.Client
	Recorder record.EventRecorder

	// Namespace is where workers, claims and pods live.
	Namespace string
	// DeviceClass is the DRA DeviceClass generated claims request.
	DeviceClass string
	// DeviceDriver is the driver publishing the ResourceSlices, and the CEL
	// domain its capacities live under. Distinct from DeviceClass in principle,
	// identical for NVIDIA's driver.
	DeviceDriver string
	// DefaultPodTemplates names the ConfigMap per role used when a worker does
	// not name one itself.
	DefaultPodTemplates map[openrlv1alpha1.WorkerRole]string
	// RetryInterval is how often a worker that could not be placed is retried.
	RetryInterval time.Duration
	// PlacementTimeout is how long a worker may go unplaced before the request
	// is declared unsatisfiable. Without it an impossible request waits
	// forever, indistinguishable from one that is merely queued.
	PlacementTimeout time.Duration
	// ReclaimInterval is how often idle claims are swept.
	ReclaimInterval time.Duration

	// reader reads straight from the API server, past the informer cache.
	//
	// A placement is three writes -- create claim, patch status, create pod --
	// and the cache reflects none of them immediately. Reading workers and
	// claims through this instead means a burst of workers reconciled back to
	// back each sees the previous one's booking, with no bookkeeping to keep in
	// sync. The reconciler is singleton and namespaced, so the extra API reads
	// are bounded. Nil (in tests) falls back to the regular client.
	reader client.Reader
}

// fleetReader is the consistent reader for fleet state.
func (r *OpenRLWorkerReconciler) fleetReader() client.Reader {
	if r.reader != nil {
		return r.reader
	}
	return r.Client
}

// +kubebuilder:rbac:groups=openrl.io,resources=openrlworkers,verbs=get;list;watch
// +kubebuilder:rbac:groups=openrl.io,resources=openrlworkers/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceclaims,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=resource.k8s.io,resources=resourceslices,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups=core,resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=events,verbs=create;patch
// +kubebuilder:rbac:groups=coordination.k8s.io,resources=leases,verbs=get;list;watch;create;update;patch;delete

// Reconcile places one worker, deciding against a fresh read of the fleet.
func (r *OpenRLWorkerReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	// The worker itself comes through the consistent reader too: its status is
	// this controller's own record of the assignment, and deciding against a
	// stale copy of your own decision is how seats get handed out twice.
	var worker openrlv1alpha1.OpenRLWorker
	if err := r.fleetReader().Get(ctx, req.NamespacedName, &worker); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}
	if !worker.DeletionTimestamp.IsZero() {
		// Pods are garbage-collected through their owner reference. Claims are
		// not owned by any single worker -- see reclaimIdleClaims.
		return ctrl.Result{}, nil
	}

	request := requestFrom(&worker)

	var workers openrlv1alpha1.OpenRLWorkerList
	if err := r.fleetReader().List(ctx, &workers, client.InNamespace(r.Namespace)); err != nil {
		return ctrl.Result{}, fmt.Errorf("list workers: %w", err)
	}
	fleet, err := r.readFleet(ctx, workers.Items)
	if err != nil {
		// Returning the error hands retry timing to the workqueue's
		// exponential backoff instead of a fixed-interval poll.
		return ctrl.Result{}, fmt.Errorf("cannot read the fleet, placing nothing this pass: %w", err)
	}

	return r.place(ctx, &worker, request, fleet)
}

func (r *OpenRLWorkerReconciler) place(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, request placement.Request, fleet *placement.Fleet) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	podName, err := workerPodName(worker.Spec.ModelID, worker.Spec.Role)
	if err != nil {
		return ctrl.Result{}, r.fail(ctx, worker, "InvalidSpec", err.Error())
	}

	pod, err := r.findPod(ctx, podName)
	if err != nil {
		return ctrl.Result{}, err
	}

	claimName := worker.Status.ClaimName
	if pod != nil && pod.Labels[LabelClaim] != "" && pod.Labels[LabelClaim] != claimName {
		// The running pod is the truth about where this worker actually is.
		// Adopting it is how the controller recovers from a restart that lost
		// an unpatched status.
		claimName = pod.Labels[LabelClaim]
	}
	if claimName != "" {
		if _, live := fleet.Claims[claimName]; !live {
			logger.Info("assigned claim no longer exists; re-placing", "claim", claimName, "worker", worker.Name)
			claimName = ""
		}
	}

	if claimName == "" {
		selection, created, err := r.assign(ctx, worker, request, fleet)
		if err != nil {
			return ctrl.Result{}, err
		}
		if selection == nil {
			reason := placement.Explain(request, fleet, "")
			if r.expired(worker) {
				return ctrl.Result{}, r.fail(ctx, worker, "Unsatisfiable", reason)
			}
			return ctrl.Result{RequeueAfter: r.RetryInterval}, r.markPending(ctx, worker, reason)
		}
		claimName = selection.Claim.Name
		perDevice := selection.PerDeviceBytes

		verb := "SharedExistingClaim"
		if created {
			verb = fmt.Sprintf("CreatedClaim: %dx%s", selection.Claim.DeviceCount, gibQuantity(perDevice))
		}
		r.warnIfExpensive(worker, request)
		if err := r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.Reason = verb
			recordFootprint(s, request, selection.Claim.DeviceCount, perDevice)
		}); err != nil {
			return ctrl.Result{}, err
		}
	}

	if pod != nil && pod.Labels[LabelClaim] != "" && pod.Labels[LabelClaim] != claimName {
		// Re-placed onto a different claim. A pod's spec.resourceClaims is
		// immutable, so this pod can never reach the new claim -- it would sit
		// unschedulable on "cannot allocate all claims" until the deadline failed
		// it, while the claim just booked for it went unused. Delete it and let
		// the next pass build one against the claim the worker now holds.
		logger.Info("pod is bound to a stale claim; recreating it", "pod", podName, "was", pod.Labels[LabelClaim], "now", claimName, "worker", worker.Name)
		if err := r.Delete(ctx, pod); err != nil && !apierrors.IsNotFound(err) {
			return ctrl.Result{}, err
		}
		// No requeue: the pod is Owned, so its deletion event re-enqueues this
		// worker the moment it lands.
		return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.Reason = "RecreatingPodOnNewClaim"
		})
	}

	if pod == nil {
		if err := r.createPod(ctx, worker, podName, claimName); err != nil {
			return ctrl.Result{}, err
		}
		// No requeue: pod events arrive through the Owns watch.
		return ctrl.Result{}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
			s.Phase = openrlv1alpha1.PhasePlacing
			s.ClaimName = claimName
			s.PodName = podName
			s.Reason = "PodCreated"
		})
	}

	if detail := unschedulableMessage(pod); detail != "" {
		// A full fleet and a fleet that is too small look identical from here.
		// Explain tells them apart; the scheduler's own words name what failed.
		reason := placement.Explain(request, fleet, detail)
		if r.expired(worker) {
			return ctrl.Result{}, r.fail(ctx, worker, "Unschedulable", reason)
		}
		return ctrl.Result{RequeueAfter: r.RetryInterval}, r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
			s.Phase = openrlv1alpha1.PhasePending
			s.ClaimName, s.PodName, s.Reason = claimName, podName, reason
			setCondition(s, metav1.ConditionFalse, "Unschedulable", reason)
		})
	}

	return ctrl.Result{}, r.reportPod(ctx, worker, fleet, pod, claimName, podName)
}

// assign joins a claim, or cuts a new one. The returned bool says which.
func (r *OpenRLWorkerReconciler) assign(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, request placement.Request, fleet *placement.Fleet) (*placement.Selection, bool, error) {
	if selection := placement.SelectClaim(request, fleet); selection != nil {
		// Book immediately, so a worker reconciled straight after this one sees
		// the seat taken.
		selection.Claim.Book(selection.Cohort, selection.PerDeviceBytes)
		return selection, false, nil
	}

	pool := placement.ChoosePool(request, fleet)
	if pool == nil {
		return nil, false, nil
	}

	perDevice := request.PerDeviceBytes(pool.DeviceCount)
	claim := &placement.Claim{
		Name:        claimNameFor(worker),
		DeviceCount: pool.DeviceCount,
		// Node stays empty: unallocated, so any worker that joins it accepts the
		// most restrictive pool it could land on.
	}
	cohort := request.CohortKey()
	claim.Book(cohort, perDevice)

	log.FromContext(ctx).Info("cutting a claim",
		"claim", claim.Name, "devices", pool.DeviceCount, "perDevice", gibQuantity(perDevice), "sizedAgainst", pool.Node.Name)

	body := r.buildClaim(worker, claim, perDevice)
	if err := r.Create(ctx, body); err != nil && !apierrors.IsAlreadyExists(err) {
		return nil, false, fmt.Errorf("create claim %s: %w", claim.Name, err)
	}
	fleet.Claims[claim.Name] = claim
	return &placement.Selection{Claim: claim, Cohort: cohort, PerDeviceBytes: perDevice}, true, nil
}

func (r *OpenRLWorkerReconciler) createPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, podName, claimName string) error {
	pod, err := r.renderPod(ctx, worker, podName, claimName)
	if err != nil {
		// Record the failure, then still return the error: fail() returning
		// nil (a successful status patch) must not read as "pod created", or
		// the caller's next status write stomps Failed with PodCreated and,
		// with no error to back off on, nothing ever retries the render.
		if patchErr := r.fail(ctx, worker, "TemplateError", err.Error()); patchErr != nil {
			return patchErr
		}
		return err
	}
	if err := r.Create(ctx, pod); err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("create pod %s: %w", podName, err)
	}
	return nil
}

// findPod returns the worker's pod, or nil if it has none.
//
// A terminal pod is returned like any other: it is reported, not replaced.
// Whether a finished model still wants a worker is the gateway's call, and
// recreating the pod here would fight it.
func (r *OpenRLWorkerReconciler) findPod(ctx context.Context, podName string) (*corev1.Pod, error) {
	var pod corev1.Pod
	err := r.Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: podName}, &pod)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read pod %s: %w", podName, err)
	}
	return &pod, nil
}

func (r *OpenRLWorkerReconciler) reportPod(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, fleet *placement.Fleet, pod *corev1.Pod, claimName, podName string) error {
	phase := openrlv1alpha1.PhasePlacing
	reason := ""
	switch pod.Status.Phase {
	case corev1.PodRunning, corev1.PodSucceeded:
		phase = openrlv1alpha1.PhaseRunning
	case corev1.PodFailed:
		phase, reason = openrlv1alpha1.PhaseFailed, "PodFailed"
	}

	node := pod.Spec.NodeName
	if node == "" {
		if claim, ok := fleet.Claims[claimName]; ok {
			node = claim.Node
		}
	}

	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
		s.Phase, s.ClaimName, s.PodName, s.NodeName, s.Reason = phase, claimName, podName, node, reason
		if phase == openrlv1alpha1.PhaseRunning {
			setCondition(s, metav1.ConditionTrue, "Placed", "worker is running on "+claimName)
		}
	})
}

// recordFootprint writes down the memory and switch cost this placement
// implies. Called from inside a patchStatus mutation, so the change detector
// sees these fields rather than mistaking an already-mutated status for an
// unchanged one.
//
// Parking a worker costs its whole accelerator footprint in host RAM, whatever
// it is spread over, so the parked figure is the request's memory itself.
func recordFootprint(status *openrlv1alpha1.OpenRLWorkerStatus, request placement.Request, deviceCount int, perDevice int64) {
	status.DeviceCount = int32(deviceCount)
	status.MemoryPerDevice = gibQuantity(perDevice)
	status.HostMemoryPerResident = gibQuantity(request.Memory)
	status.EstimatedSwitchTime = switchCost(request).String()
}

// switchCost is how long the timeslicer is expected to take to park this worker
// and restore the next one, rounded to something worth printing.
func switchCost(request placement.Request) time.Duration {
	return placement.SwitchCost(request.Memory).Round(100 * time.Millisecond)
}

// warnIfExpensive complains when parking this worker costs enough that sharing
// its claim is a real tax rather than a free win.
//
// The figure is this worker's own, not its cohort's: a cohort it joins later
// may be larger, but what the operator can act on is the worker in front of
// them.
func (r *OpenRLWorkerReconciler) warnIfExpensive(worker *openrlv1alpha1.OpenRLWorker, request placement.Request) {
	cost := switchCost(request)
	if cost < SwitchCostWarnThreshold || r.Recorder == nil {
		return
	}
	r.Recorder.Eventf(worker, corev1.EventTypeWarning, "ExpensiveTimeSlice",
		"parking this worker moves %s to host memory, about %s per context switch; "+
			"consider openrl.io/max-residents=1 on pools that host it",
		gibQuantity(request.Memory), cost)
}

// -- status -------------------------------------------------------------------

func (r *OpenRLWorkerReconciler) patchStatus(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, mutate func(*openrlv1alpha1.OpenRLWorkerStatus)) error {
	before := worker.Status.DeepCopy()
	mutate(&worker.Status)
	worker.Status.ObservedGeneration = worker.Generation
	// DeepEqual is safe against LastTransitionTime churn because setCondition
	// only moves it when the condition's status actually flips.
	if apiequality.Semantic.DeepEqual(before, &worker.Status) {
		return nil
	}
	if err := r.Status().Update(ctx, worker); err != nil {
		if apierrors.IsConflict(err) {
			// Someone else wrote first; the next reconcile recomputes from scratch.
			return nil
		}
		return fmt.Errorf("patch status of %s: %w", worker.Name, err)
	}
	return nil
}

func (r *OpenRLWorkerReconciler) markPending(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, reason string) error {
	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
		s.Phase, s.Reason = openrlv1alpha1.PhasePending, reason
		setCondition(s, metav1.ConditionFalse, "WaitingForCapacity", reason)
	})
}

func (r *OpenRLWorkerReconciler) fail(ctx context.Context, worker *openrlv1alpha1.OpenRLWorker, reason, message string) error {
	if r.Recorder != nil {
		r.Recorder.Event(worker, corev1.EventTypeWarning, reason, message)
	}
	return r.patchStatus(ctx, worker, func(s *openrlv1alpha1.OpenRLWorkerStatus) {
		s.Phase, s.Reason = openrlv1alpha1.PhaseFailed, message
		setCondition(s, metav1.ConditionFalse, reason, message)
	})
}

// expired reports whether this worker has been waiting past the point where
// "not yet" should be called "no".
func (r *OpenRLWorkerReconciler) expired(worker *openrlv1alpha1.OpenRLWorker) bool {
	if r.PlacementTimeout <= 0 {
		return false
	}
	since := worker.CreationTimestamp.Time
	if condition := apimeta.FindStatusCondition(worker.Status.Conditions, openrlv1alpha1.ConditionPlaced); condition != nil {
		since = condition.LastTransitionTime.Time
	}
	return time.Since(since) > r.PlacementTimeout
}

func setCondition(status *openrlv1alpha1.OpenRLWorkerStatus, state metav1.ConditionStatus, reason, message string) {
	// Kubernetes rejects condition messages over 32KiB; SetStatusCondition
	// does not truncate.
	if len(message) > 32768 {
		message = message[:32768]
	}
	apimeta.SetStatusCondition(&status.Conditions, metav1.Condition{
		Type:    openrlv1alpha1.ConditionPlaced,
		Status:  state,
		Reason:  reason,
		Message: message,
	})
}

// -- wiring --------------------------------------------------------------------

// SetupWithManager registers the reconciler and the claim-reclaim sweep.
func (r *OpenRLWorkerReconciler) SetupWithManager(mgr ctrl.Manager) error {
	r.reader = mgr.GetAPIReader()

	if err := mgr.Add(manager.RunnableFunc(r.runReclaim)); err != nil {
		return err
	}

	// Capacity changes are fleet-wide, so a claim being allocated or a node
	// being labelled wakes every worker rather than one.
	wakeAll := handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, _ client.Object) []reconcile.Request {
		var workers openrlv1alpha1.OpenRLWorkerList
		if err := mgr.GetClient().List(ctx, &workers, client.InNamespace(r.Namespace)); err != nil {
			return nil
		}
		requests := make([]reconcile.Request, 0, len(workers.Items))
		for i := range workers.Items {
			requests = append(requests, reconcile.Request{
				NamespacedName: types.NamespacedName{Namespace: workers.Items[i].Namespace, Name: workers.Items[i].Name},
			})
		}
		return requests
	})

	// Kubelet heartbeats rewrite node status every few seconds; without a
	// predicate each one would fan out into a reconcile per worker, forever.
	// Placement only reads node labels and allocatable memory, so only those
	// changes (and nodes appearing or leaving) are capacity events.
	nodeCapacityChanged := predicate.Or(
		predicate.LabelChangedPredicate{},
		predicate.Funcs{
			UpdateFunc: func(e event.UpdateEvent) bool {
				before, okBefore := e.ObjectOld.(*corev1.Node)
				after, okAfter := e.ObjectNew.(*corev1.Node)
				return okBefore && okAfter && !before.Status.Allocatable.Memory().Equal(*after.Status.Allocatable.Memory())
			},
		},
	)

	return ctrl.NewControllerManagedBy(mgr).
		For(&openrlv1alpha1.OpenRLWorker{}).
		Owns(&corev1.Pod{}).
		Watches(&resourcev1.ResourceClaim{}, wakeAll, builder.WithPredicates(managedClaims())).
		Watches(&corev1.Node{}, wakeAll, builder.WithPredicates(nodeCapacityChanged)).
		WithOptions(controller.Options{MaxConcurrentReconciles: 1}).
		Complete(r)
}
