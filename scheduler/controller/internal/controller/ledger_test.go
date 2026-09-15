package controller

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

func getLedger(t *testing.T, r *WorkloadReconciler, name string) *openrlv1alpha1.ClaimLedger {
	t.Helper()
	var ledger openrlv1alpha1.ClaimLedger
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, &ledger); err != nil {
		t.Fatalf("get ledger %s: %v", name, err)
	}
	return &ledger
}

// expectGone asserts the object was deleted: retirement is proven by
// absence, not by an empty spec.
func expectGone(t *testing.T, r *WorkloadReconciler, obj client.Object, name string) {
	t.Helper()
	err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: name}, obj)
	if !apierrors.IsNotFound(err) {
		t.Fatalf("%T %q still present: %v", obj, name, err)
	}
}

// Placing a worker writes the same booking in three places: the ledger's
// seat, the worker's status, and the pod's env. All three must agree, or
// the runtime gate has nothing to check.
func TestPlacingRecordsTheSeatEverywhere(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")

	w := getWorker(t, r, "w-a")
	if w.Status.AssignmentID == "" {
		t.Fatal("status carries no assignment")
	}

	ledger := getLedger(t, r, ledgerNameFor(w.Status.ClaimName))
	if ledger.Spec.ClaimName != w.Status.ClaimName {
		t.Errorf("ledger pairs with claim %q, worker holds %q", ledger.Spec.ClaimName, w.Status.ClaimName)
	}
	seat := findSeat(ledger, "w-a")
	if seat == nil {
		t.Fatalf("ledger %s records no seat for w-a: %+v", ledger.Name, ledger.Spec.Seats)
	}
	if seat.AssignmentID != w.Status.AssignmentID {
		t.Errorf("seat assignment %q, status says %q", seat.AssignmentID, w.Status.AssignmentID)
	}

	pod := getPod(t, r, "orw-w-a")
	if pod == nil {
		t.Fatal("no pod for w-a")
	}
	if got := envOf(pod.Spec.Containers[0], claimLedgerEnv); got != ledger.Name {
		t.Errorf("pod %s=%q, want %q", claimLedgerEnv, got, ledger.Name)
	}
	if got := envOf(pod.Spec.Containers[0], assignmentIDEnv); got != w.Status.AssignmentID {
		t.Errorf("pod %s=%q, want %q", assignmentIDEnv, got, w.Status.AssignmentID)
	}
}

// A joiner books a seat beside the founder's on the founder's ledger; it does
// not get a ledger of its own.
func TestJoinerBooksASeatNextToTheFounder(t *testing.T) {
	r := newReconciler(t, append(enabledNode(),
		trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"), trainerWorker("w-c", "model-c"))...)

	runReconcile(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	runReconcile(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))
	runReconcile(t, r, "w-c")
	abandoned := ledgerNameFor(claimOf(t, r, "w-c"))
	fallBackToSharing(t, r, "w-c")

	shared := claimOf(t, r, "w-c")
	ledger := getLedger(t, r, ledgerNameFor(shared))
	if len(ledger.Spec.Seats) != 2 {
		t.Fatalf("shared ledger holds %d seats, want 2: %+v", len(ledger.Spec.Seats), ledger.Spec.Seats)
	}
	if findSeat(ledger, "w-c") == nil {
		t.Errorf("no seat for the joiner w-c: %+v", ledger.Spec.Seats)
	}
	// Releasing the founder-less ledger's last seat retires it inline; only
	// its absence proves the seat came back.
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, abandoned)
}

// Booking is idempotent per incarnation: ensuring the same worker's seat
// again adopts the recorded assignment instead of minting a second one.
func TestRebookingAdoptsTheRecordedSeat(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	w := getWorker(t, r, "w-a")
	first := w.Status.AssignmentID

	_, seat, err := r.ensureSeat(context.Background(), w.Status.ClaimName, newSeat(w, requestFrom(w)), true)
	if err != nil {
		t.Fatal(err)
	}
	if seat.AssignmentID != first {
		t.Errorf("rebooking replaced assignment %q with %q", first, seat.AssignmentID)
	}
	ledger := getLedger(t, r, ledgerNameFor(w.Status.ClaimName))
	if len(ledger.Spec.Seats) != 1 {
		t.Fatalf("rebooking grew the chart to %d seats: %+v", len(ledger.Spec.Seats), ledger.Spec.Seats)
	}
}

func TestSharingRechecksTheLedgerAfterSelection(t *testing.T) {
	resident, incoming := trainerWorker("resident", "model-a"), trainerWorker("incoming", "model-b")
	r := newReconciler(t, append(enabledNode(), resident, incoming)...)
	settle(t, r, resident.Name)
	claim := claimOf(t, r, resident.Name)
	allocateClaim(t, r, claim)
	fleet, err := r.readFleet(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	request := requestFrom(incoming)
	if placement.SelectClaim(request, fleet) == nil {
		t.Fatal("snapshot should offer the FFT claim")
	}
	// The current ledger now holds an exclusive worker, while the informer
	// snapshot still offers a shareable claim. The consistent booking must refuse it.
	ledger := getLedger(t, r, ledgerNameFor(claim))
	ledger.Spec.Seats[0].Exclusive = true
	if err := r.Update(context.Background(), ledger); err != nil {
		t.Fatal(err)
	}
	joined, _, err := r.joinExistingClaim(context.Background(), incoming, request, fleet, "")
	if err != nil || joined != nil {
		t.Fatalf("stale selection joined=%v, error=%v", joined, err)
	}
	if got := len(getLedger(t, r, ledger.Name).Spec.Seats); got != 1 {
		t.Fatalf("refused join changed occupancy to %d seats", got)
	}
}

type conflictOnceClient struct {
	client.Client
	onConflict func()
	fired      bool
}

func (c *conflictOnceClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	if _, ok := obj.(*openrlv1alpha1.ClaimLedger); ok && !c.fired {
		c.fired = true
		c.onConflict()
		return apierrors.NewConflict(openrlv1alpha1.GroupVersion.WithResource("claimledgers").GroupResource(), obj.GetName(), fmt.Errorf("concurrent seat booking"))
	}
	return c.Client.Update(ctx, obj, opts...)
}

// Two workers reconciled close together can both see the node as fitting in
// their snapshots. When the second books against the updated ledger, post-booking
// fleet verification sees the node is over capacity and releases the seat back.
func TestSharingRechecksHostMemoryAfterSelection(t *testing.T) {
	resident := hungryWorker("resident", "model-a", "200Gi")
	first := hungryWorker("w-b", "model-b", "100Gi")
	second := hungryWorker("w-c", "model-c", "100Gi")
	r := newReconciler(t, append(enabledNode(), resident, first, second)...)
	settle(t, r, resident.Name)
	claim := claimOf(t, r, resident.Name)
	allocateClaim(t, r, claim)

	// Both workers read the fleet before either books: 200Gi + 100Gi = 300Gi
	// fits the 340Gi node in both snapshots.
	fleetB, err := r.readFleet(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	fleetC, err := r.readFleet(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	// Simulate a true CAS conflict: w-c reads the ledger before w-b commits,
	// w-b commits first causing w-c's Update to conflict, w-c retries and writes
	// its seat, and joinExistingClaim's post-booking readFleet sees 400Gi > 340Gi
	// and releases w-c's seat back.
	baseClient := r.Client
	r.Client = &conflictOnceClient{
		Client: baseClient,
		onConflict: func() {
			rB := *r
			rB.Client = baseClient
			joinedB, _, err := rB.joinExistingClaim(context.Background(), first, requestFrom(first), fleetB, "")
			if err != nil || joinedB == nil {
				t.Fatalf("concurrent first join failed: joined=%v, err=%v", joinedB, err)
			}
		},
	}

	joinedC, _, err := r.joinExistingClaim(context.Background(), second, requestFrom(second), fleetC, "")
	if err != nil || joinedC != nil {
		t.Fatalf("second join after CAS conflict should be refused on host memory: joined=%v, err=%v", joinedC, err)
	}
	if got := len(getLedger(t, r, ledgerNameFor(claim)).Spec.Seats); got != 2 {
		t.Fatalf("ledger holds %d seats, want 2 (resident + w-b)", got)
	}
}

// Two workers racing onto two different claims on the same node never touch
// the same ledger, so neither CAS conflicts. Re-reading the fleet after booking
// catches the cross-claim node overcommit and releases the seat back.
func TestSharingCrossClaimRaceReleasesOvercommittedSeat(t *testing.T) {
	// testNode has 2 GPUs and 340Gi allocatable memory.
	wA := hungryWorker("w-a", "model-a", "100Gi")
	wB := hungryWorker("w-b", "model-b", "100Gi")
	wZero := trainerWorker("w-zero", "model-z") // 0Gi host request
	wC := hungryWorker("w-c", "model-c", "100Gi")
	wD := hungryWorker("w-d", "model-d", "100Gi")
	r := newReconciler(t, append(enabledNode(), wA, wB, wZero, wC, wD)...)

	settle(t, r, wA.Name)
	claimA := claimOf(t, r, wA.Name)
	allocateClaim(t, r, claimA)

	settle(t, r, wB.Name)
	claimB := claimOf(t, r, wB.Name)
	allocateClaim(t, r, claimB)

	firstClaim, secondClaim := claimA, claimB
	if claimB < claimA {
		firstClaim, secondClaim = claimB, claimA
	}

	// fleetC sees 1 worker on each claim (200Gi total); SelectClaim picks firstClaim.
	fleetC, err := r.readFleet(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	// Seat a 0Gi worker on firstClaim so fleetD sees 2 workers on firstClaim
	// and 1 worker on secondClaim while node host memory is still 200Gi / 340Gi.
	// SelectClaim on fleetD therefore picks secondClaim.
	if _, _, err := r.ensureSeat(context.Background(), firstClaim, newSeat(wZero, requestFrom(wZero)), false); err != nil {
		t.Fatal(err)
	}
	fleetD, err := r.readFleet(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	// w-c joins firstClaim (brings node to 300Gi / 340Gi).
	joinedC, _, err := r.joinExistingClaim(context.Background(), wC, requestFrom(wC), fleetC, "")
	if err != nil || joinedC == nil || joinedC.Name != firstClaim {
		t.Fatalf("w-c join onto firstClaim failed: joined=%v, err=%v", joinedC, err)
	}

	// w-d races onto secondClaim using pre-booking snapshot fleetD.
	// Because secondClaim is a different ledger, its Update succeeds without a
	// CAS conflict, but joinExistingClaim's post-booking readFleet sees
	// 400Gi > 340Gi across both claims on the node and releases w-d's seat.
	joinedD, _, err := r.joinExistingClaim(context.Background(), wD, requestFrom(wD), fleetD, "")
	if err != nil || joinedD != nil {
		t.Fatalf("w-d cross-claim join should be refused and rolled back: joined=%v, err=%v", joinedD, err)
	}

	// Verify w-d's seat was released from secondClaim's ledger, leaving only its founder.
	if got := len(getLedger(t, r, ledgerNameFor(secondClaim)).Spec.Seats); got != 1 {
		t.Fatalf("secondClaim ledger holds %d seats after rollback, want 1", got)
	}
}

// Deleting a worker frees its seat only once the pod is verifiably gone, and
// the same reconcile then retires the empty ledger and its claim.
func TestTeardownReclaimsTheClaimAndGroupInline(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	claimName := getWorker(t, r, "w-a").Status.ClaimName
	ledgerName := ledgerNameFor(claimName)

	if err := r.Delete(context.Background(), getWorker(t, r, "w-a")); err != nil {
		t.Fatal(err)
	}
	// First pass deletes the pod; second observes it gone, frees the last
	// seat, and reclaims the ledger and claim in the same reconcile -- an
	// allocated claim pins a device, so nothing may linger.
	runReconcile(t, r, "w-a")
	runReconcile(t, r, "w-a")

	expectGone(t, r, &openrlv1alpha1.Workload{}, "w-a")
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerName)
	expectGone(t, r, &resourcev1.ResourceClaim{}, claimName)
}

// A dedicated claim that DRA has not satisfied is abandoned the moment
// kube-scheduler declines its pod -- the verdict, not a timer, is the
// trigger: the worker books a seat on an allocated claim, frees its old
// seat, and its pod is rebuilt against the shared claim.
func TestPendingWorkerFallsBackToSharingOnTheVerdict(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	// w-a's claim allocates; w-b's never does -- the device DRA priced it
	// for went to someone else.
	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")
	dedicated := claimOf(t, r, "w-b")

	// Without the verdict the worker waits on its dedicated claim: a pod
	// kube-scheduler has not judged yet is not stuck.
	runReconcile(t, r, "w-b")
	if got := claimOf(t, r, "w-b"); got != dedicated {
		t.Fatalf("w-b moved to %q without kube-scheduler's verdict", got)
	}

	markUnschedulable(t, r, "w-b", time.Now())

	runReconcile(t, r, "w-b")
	shared := claimOf(t, r, "w-a")
	if got := claimOf(t, r, "w-b"); got != shared {
		t.Fatalf("w-b holds %q after the verdict, want the allocated claim %q", got, shared)
	}
	if seat := findSeat(getLedger(t, r, ledgerNameFor(shared)), "w-b"); seat == nil {
		t.Fatal("no seat for w-b on the shared ledger")
	}
	// The abandoned dedicated claim and its ledger are reclaimed in the same
	// reconcile, retracting the scale-out signal the moment we stop wanting it.
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerNameFor(dedicated))
	expectGone(t, r, &resourcev1.ResourceClaim{}, dedicated)

	// The old pod was bound to the dedicated claim; the swap rebuilds it.
	runReconcile(t, r, "w-b")
	pod := getPod(t, r, "orw-w-b")
	if pod == nil {
		t.Fatal("no pod for w-b after the move")
	}
	if got := pod.Labels[LabelClaim]; got != shared {
		t.Errorf("pod bound to %q, want the shared claim %q", got, shared)
	}
}

// An allocated claim whose node refuses the pod past the wedge grace is
// abandoned: pod, seat, and claim all go, and the worker starts its
// lifecycle over.
func TestAbandonsAWedgedAllocatedClaim(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	wedged := claimOf(t, r, "w-a")
	allocateClaim(t, r, wedged)
	// Host memory vanished between pod incarnations (or the Spot node did):
	// kube-scheduler has refused the pod since well past the wedge grace.
	markUnschedulable(t, r, "w-a", time.Now().Add(-2*wedgeGracePeriod))

	runReconcile(t, r, "w-a")

	if got := getWorker(t, r, "w-a").Status.ClaimName; got != "" {
		t.Fatalf("worker still holds %q, want the wedged claim abandoned", got)
	}
	if pod := getPod(t, r, "orw-w-a"); pod != nil {
		t.Fatal("the wedged pod survived")
	}
	expectGone(t, r, &resourcev1.ResourceClaim{}, wedged)
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerNameFor(wedged))

	// The lifecycle restarts cleanly: fresh claim, fresh seat.
	settle(t, r, "w-a")
	fresh := claimOf(t, r, "w-a")
	if findSeat(getLedger(t, r, ledgerNameFor(fresh)), "w-a") == nil {
		t.Fatal("no seat after the restart")
	}
}

// A claim deleted out from under its worker leaves the ledger holding a seat
// nothing tracks. Re-placing must release that booking, not adopt it: claim
// names are deterministic, so the proof is a fresh assignment ID on a fresh
// ledger, never the stale seat carried over.
func TestVanishedClaimReleasesItsSeat(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"))...)

	settle(t, r, "w-a")
	claimName := claimOf(t, r, "w-a")
	first := getWorker(t, r, "w-a").Status.AssignmentID
	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: claimName}, &claim); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), &claim); err != nil {
		t.Fatal(err)
	}

	settle(t, r, "w-a")
	if got := getWorker(t, r, "w-a").Status.AssignmentID; got == first {
		t.Fatal("the stale seat was adopted; want it released and re-booked")
	}
	ledger := getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-a")))
	if len(ledger.Spec.Seats) != 1 {
		t.Fatalf("ledger holds %d seats after the re-place, want 1: %+v", len(ledger.Spec.Seats), ledger.Spec.Seats)
	}
}

// One seat per worker: a booking stranded on another ledger -- a status write
// lost after a seat move -- is released by the holder's next reconcile.
func TestStraySeatIsReleasedOnReconcile(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)

	settle(t, r, "w-a")
	allocateClaim(t, r, claimOf(t, r, "w-a"))
	settle(t, r, "w-b")
	allocateClaim(t, r, claimOf(t, r, "w-b"))

	// Strand a seat: book w-a onto w-b's ledger behind its status's back.
	wa := getWorker(t, r, "w-a")
	if _, _, err := r.ensureSeat(context.Background(), claimOf(t, r, "w-b"), newSeat(wa, requestFrom(wa)), false); err != nil {
		t.Fatal(err)
	}

	settle(t, r, "w-a")
	ledger := getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-b")))
	if findSeat(ledger, "w-a") != nil {
		t.Errorf("stray seat for w-a survived its reconcile: %+v", ledger.Spec.Seats)
	}
	if findSeat(ledger, "w-b") == nil {
		t.Error("the release took the rightful tenant's seat with it")
	}
	if findSeat(getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-a"))), "w-a") == nil {
		t.Error("w-a lost its own seat")
	}
}

// A retiring claim -- ledger already deleted, claim not yet -- must never be
// re-seated: joins append to existing ledgers only, so the ledger's absence is
// the tombstone, and no retiring state is needed between empty and gone.
func TestJoinNeverResurrectsARetiringGroup(t *testing.T) {
	orphan := &resourcev1.ResourceClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name: "claim-orphan", Namespace: testNamespace,
			Labels: map[string]string{LabelManaged: "true"},
		},
		Status: resourcev1.ResourceClaimStatus{
			Allocation: &resourcev1.AllocationResult{
				Devices: resourcev1.DeviceAllocationResult{
					Results: []resourcev1.DeviceRequestAllocationResult{{
						Request: podClaimName, Driver: testDriver, Pool: testNode, Device: "gpu-0",
					}},
				},
				NodeSelector: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{{
					MatchFields: []corev1.NodeSelectorRequirement{{
						Key: "metadata.name", Operator: corev1.NodeSelectorOpIn, Values: []string{testNode},
					}},
				}}},
			},
		},
	}
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), orphan)...)
	r.PlacementStrategy = placement.StrategyBinPack

	// Binpack would love the orphan: allocated, empty, fewest owners. The
	// missing ledger must turn the join away and force a dedicated claim.
	settle(t, r, "w-a")
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerNameFor("claim-orphan"))
	if got := claimOf(t, r, "w-a"); got == "claim-orphan" {
		t.Fatal("worker seated on a retiring claim")
	}
}

// An earlier workload with the same name, deleted before its own repair
// pass, can leave a seat under our name with its UID. The stray release is
// by name, so our next reconcile removes it and it stops counting against
// the node.
func TestPredecessorsStraySeatIsReleased(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)
	settle(t, r, "w-a")
	settle(t, r, "w-b")

	ledger := getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-b")))
	ledger.Spec.Seats = append(ledger.Spec.Seats, openrlv1alpha1.Seat{Workload: "w-a", WorkloadUID: "old-workload", AssignmentID: "stale"})
	if err := r.Update(context.Background(), ledger); err != nil {
		t.Fatal(err)
	}

	runReconcile(t, r, "w-a")
	ledger = getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-b")))
	if findSeat(ledger, "w-a") != nil {
		t.Errorf("the old workload's seat survived our reconcile: %+v", ledger.Spec.Seats)
	}
	if findSeat(ledger, "w-b") == nil {
		t.Error("the release took the rightful tenant's seat with it")
	}
	if findSeat(getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-a"))), "w-a") == nil {
		t.Error("w-a lost its own seat")
	}
}

// Teardown releases every seat under the name, not just the one the status
// records: a booking a lost status write never recorded would otherwise
// outlive its workload, with nothing left to release it.
func TestTeardownReleasesSeatsTheStatusForgot(t *testing.T) {
	r := newReconciler(t, append(enabledNode(), trainerWorker("w-a", "model-a"), trainerWorker("w-b", "model-b"))...)
	settle(t, r, "w-a")
	settle(t, r, "w-b")

	// Strand a seat on w-b's ledger behind w-a's status, then delete w-a
	// with no pod left to wait for.
	wa := getWorker(t, r, "w-a")
	if _, _, err := r.ensureSeat(context.Background(), claimOf(t, r, "w-b"), newSeat(wa, requestFrom(wa)), false); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), getPod(t, r, "orw-w-a")); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete(context.Background(), wa); err != nil {
		t.Fatal(err)
	}
	runReconcile(t, r, "w-a")

	expectGone(t, r, &openrlv1alpha1.Workload{}, "w-a")
	ledger := getLedger(t, r, ledgerNameFor(claimOf(t, r, "w-b")))
	if findSeat(ledger, "w-a") != nil {
		t.Errorf("the stranded seat outlived its workload: %+v", ledger.Spec.Seats)
	}
	if findSeat(ledger, "w-b") == nil {
		t.Error("the release took the rightful tenant's seat with it")
	}
}

// A worker re-cutting its dedicated claim while DRA's finalizer still holds
// the one it abandoned waits, and the wait is named for what it is: not a
// capacity verdict, so the placement clock does not run, and no ledger is
// seated on a claim that does not exist.
func TestWaitsForATerminatingPredecessorClaim(t *testing.T) {
	w := trainerWorker("w-a", "model-a")
	w.UID = "11111111-aaaa-bbbb-cccc-dddddddddddd"
	w.CreationTimestamp = metav1.NewTime(time.Now().Add(-time.Hour))
	dying := &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{
		Name: claimNameFor(w), Namespace: testNamespace,
		Labels:     map[string]string{LabelManaged: "true"},
		Finalizers: []string{"resource.kubernetes.io/delete-protection"},
	}}
	r := newReconciler(t, append(enabledNode(), w, dying)...)
	r.PlacementTimeout = time.Minute
	if err := r.Delete(context.Background(), dying); err != nil {
		t.Fatal(err)
	}

	result := runReconcile(t, r, "w-a")
	if result.RequeueAfter != r.RetryInterval {
		t.Errorf("requeueAfter = %v, want %v", result.RequeueAfter, r.RetryInterval)
	}
	after := getWorker(t, r, "w-a")
	if after.Status.Phase != openrlv1alpha1.PhasePending || !strings.Contains(after.Status.Reason, "WaitingForPredecessorClaim") {
		t.Fatalf("status = %s %q, want Pending WaitingForPredecessorClaim despite the expired clock", after.Status.Phase, after.Status.Reason)
	}
	expectGone(t, r, &openrlv1alpha1.ClaimLedger{}, ledgerNameFor(claimNameFor(w)))

	// The finalizer lets go: the next passes cut the claim afresh.
	var claim resourcev1.ResourceClaim
	if err := r.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: claimNameFor(w)}, &claim); err != nil {
		t.Fatal(err)
	}
	claim.Finalizers = nil
	if err := r.Update(context.Background(), &claim); err != nil {
		t.Fatal(err)
	}
	settle(t, r, "w-a")
	if got := claimOf(t, r, "w-a"); got != claimNameFor(w) {
		t.Fatalf("placed on %q, want the fresh %q", got, claimNameFor(w))
	}
}
