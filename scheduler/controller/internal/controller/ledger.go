package controller

import (
	"context"
	"fmt"
	"strings"

	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"sigs.k8s.io/controller-runtime/pkg/client"

	openrlv1alpha1 "github.com/gke-labs/open-rl/scheduler/controller/api/v1alpha1"
	"github.com/gke-labs/open-rl/scheduler/controller/internal/placement"
)

// bookAttempts bounds the CAS retry loop; past it the reconcile requeues.
const bookAttempts = 5

// errBookingContended means the ledger no longer accepts this seat or the
// booking exhausted its CAS retries. Callers retry placement on a later pass.
var errBookingContended = fmt.Errorf("chosen ledger could not accept the seat")

// ledgerNameFor derives the ClaimLedger name from the claim name; it is never
// stored anywhere.
func ledgerNameFor(claimName string) string {
	return "ledger-" + strings.TrimPrefix(claimName, "claim-")
}

// newSeat mints a fresh assignment ID; owner and host figures come from the
// Request so they cannot drift from placement's.
func newSeat(worker *openrlv1alpha1.Workload, request placement.Request) openrlv1alpha1.Seat {
	return openrlv1alpha1.Seat{
		Workload:     worker.Name,
		WorkloadUID:  worker.UID,
		AssignmentID: string(uuid.NewUUID()),
		OwnerID:      request.OwnerKey(),
		Exclusive:    worker.Spec.Exclusive,
		HostRequest:  *resource.NewQuantity(request.HostRequestBytes, resource.BinarySI),
	}
}

// ensureSeat books the seat on the claim's ClaimLedger via one CAS loop. A seat
// held by the same incarnation is adopted and a predecessor's is replaced.
// There is no seat ceiling: host memory is the limit on parked workers,
// checked at selection and verified against a fresh fleet read after booking.
//
// createMissing is true only for a worker's own claim: the founder books its
// ClaimLedger before the claim exists, and a re-book heals the ledger under a
// running pod. A join must never create -- retirement deletes the ClaimLedger
// before the claim, so for a joiner the ClaimLedger's absence is the tombstone of
// a dying claim, and booking would resurrect it.
func (r *WorkloadReconciler) ensureSeat(ctx context.Context, claimName string, seat openrlv1alpha1.Seat, createMissing bool) (*openrlv1alpha1.ClaimLedger, *openrlv1alpha1.Seat, error) {
	ledgerName := ledgerNameFor(claimName)
	for attempt := 0; attempt < bookAttempts; attempt++ {
		var claimLedger openrlv1alpha1.ClaimLedger
		err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: ledgerName}, &claimLedger)
		if apierrors.IsNotFound(err) {
			if !createMissing {
				return nil, nil, errBookingContended
			}
			fresh := &openrlv1alpha1.ClaimLedger{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ledgerName,
					Namespace: r.Namespace,
					Labels:    map[string]string{LabelManaged: "true", LabelClaim: claimName},
				},
				Spec: openrlv1alpha1.ClaimLedgerSpec{
					ClaimName: claimName,
					Seats:     []openrlv1alpha1.Seat{seat},
				},
			}
			if err := r.Create(ctx, fresh); err != nil {
				if apierrors.IsAlreadyExists(err) {
					continue // Lost the create; book on the winner's copy.
				}
				return nil, nil, fmt.Errorf("create ledger %s: %w", ledgerName, err)
			}
			return fresh, &fresh.Spec.Seats[0], nil
		}
		if err != nil {
			return nil, nil, fmt.Errorf("read ledger %s: %w", ledgerName, err)
		}

		existing := findSeat(&claimLedger, seat.Workload)
		if existing != nil && existing.WorkloadUID == seat.WorkloadUID {
			return &claimLedger, existing, nil
		}
		// Recheck compatibility on the consistent ledger inside the CAS loop.
		// A stale fleet must not let a worker join a resident exclusive worker,
		// or let a recreated exclusive worker replace a seat in a shared claim.
		for _, other := range claimLedger.Spec.Seats {
			if other.Workload != seat.Workload && (seat.Exclusive || other.Exclusive) {
				return nil, nil, errBookingContended
			}
		}
		if existing != nil {
			*existing = seat
		} else {
			claimLedger.Spec.Seats = append(claimLedger.Spec.Seats, seat)
		}

		err = r.Update(ctx, &claimLedger)
		if err == nil {
			return &claimLedger, findSeat(&claimLedger, seat.Workload), nil
		}
		if !apierrors.IsConflict(err) {
			return nil, nil, fmt.Errorf("book seat on ledger %s: %w", ledgerName, err)
		}
		// Conflict: someone else booked first. Re-read and re-check.
	}
	return nil, nil, errBookingContended
}

// releaseSeatAndReclaim removes the seat under a workload name via one CAS
// loop. It matches by name, not UID. A seat under our name is either ours
// or was left by an earlier workload with the same name, and nothing else
// would ever release that one. A missing ledger or seat counts as released.
//
// Releasing the last seat deletes the ledger, conditioned on the version
// that showed it empty so a concurrent join wins, and then deletes the
// claim, since an allocated claim pins its GPU with or without pods. If the
// controller dies between the two deletes, the claim's owner reference lets
// GC finish.
func (r *WorkloadReconciler) releaseSeatAndReclaim(ctx context.Context, claimName, workloadName string) error {
	ledgerName := ledgerNameFor(claimName)
	for attempt := 0; attempt < bookAttempts; attempt++ {
		var claimLedger openrlv1alpha1.ClaimLedger
		if err := r.fleetReader().Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: ledgerName}, &claimLedger); err != nil {
			if apierrors.IsNotFound(err) {
				return r.deleteClaim(ctx, claimName)
			}
			return fmt.Errorf("read ledger %s: %w", ledgerName, err)
		}

		kept := claimLedger.Spec.Seats[:0]
		for _, seat := range claimLedger.Spec.Seats {
			if seat.Workload == workloadName {
				continue
			}
			kept = append(kept, seat)
		}
		if len(kept) == 0 {
			err := r.Delete(ctx, &claimLedger, client.Preconditions{ResourceVersion: &claimLedger.ResourceVersion})
			if apierrors.IsConflict(err) {
				continue // someone booked between the read and the delete
			}
			if err != nil && !apierrors.IsNotFound(err) {
				return fmt.Errorf("retire ledger %s: %w", ledgerName, err)
			}
			return r.deleteClaim(ctx, claimName)
		}
		if len(kept) == len(claimLedger.Spec.Seats) {
			return nil
		}
		claimLedger.Spec.Seats = kept

		err := r.Update(ctx, &claimLedger)
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return fmt.Errorf("release seat on ledger %s: %w", ledgerName, err)
		}
	}
	return fmt.Errorf("release seat on ledger %s: conflict retries exhausted", ledgerName)
}

// deleteClaim removes a seatless claim; already gone is success.
func (r *WorkloadReconciler) deleteClaim(ctx context.Context, claimName string) error {
	claim := &resourcev1.ResourceClaim{ObjectMeta: metav1.ObjectMeta{Name: claimName, Namespace: r.Namespace}}
	if err := r.Delete(ctx, claim); err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("reclaim claim %s: %w", claimName, err)
	}
	return nil
}

// seatedClaims lists every claim whose ledger seats this workload name. It
// reads ledgers rather than the fleet, so seats whose claim is already gone
// are included. It reads from the cache, which can miss a seat but never
// invent one, and the release re-reads consistently.
func (r *WorkloadReconciler) seatedClaims(ctx context.Context, workloadName string) ([]string, error) {
	var ledgers openrlv1alpha1.ClaimLedgerList
	if err := r.List(ctx, &ledgers, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return nil, fmt.Errorf("list claim ledgers: %w", err)
	}
	var claims []string
	for i := range ledgers.Items {
		if findSeat(&ledgers.Items[i], workloadName) != nil {
			claims = append(claims, ledgers.Items[i].Spec.ClaimName)
		}
	}
	return claims, nil
}

// findSeat returns the seat held under a workload name, or nil.
func findSeat(claimLedger *openrlv1alpha1.ClaimLedger, workload string) *openrlv1alpha1.Seat {
	for i := range claimLedger.Spec.Seats {
		if claimLedger.Spec.Seats[i].Workload == workload {
			return &claimLedger.Spec.Seats[i]
		}
	}
	return nil
}
