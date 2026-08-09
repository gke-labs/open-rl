package controller

import (
	"context"
	"time"

	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"

	openrlv1alpha1 "github.com/gke-labs/open-rl/controller/api/v1alpha1"
)

// claimGracePeriod is how long a newly cut claim is safe from the reclaim
// sweep. A claim is created before the worker status that names it, so
// without a grace period the sweep could delete a claim out from under a
// worker that is still being placed.
const claimGracePeriod = 2 * time.Minute

// managedClaims restricts the ResourceClaim watch to the ones this controller
// created, so unrelated DRA traffic does not wake every worker.
func managedClaims() predicate.Predicate {
	return predicate.NewPredicateFuncs(func(obj client.Object) bool {
		return obj.GetLabels()[LabelManaged] == "true"
	})
}

// runReclaim sweeps idle claims until the manager stops.
func (r *OpenRLWorkerReconciler) runReclaim(ctx context.Context) error {
	ticker := time.NewTicker(r.ReclaimInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return nil
		case <-ticker.C:
			if err := r.reclaimIdleClaims(ctx); err != nil {
				log.FromContext(ctx).Error(err, "reclaim sweep failed")
			}
		}
	}
}

// reclaimIdleClaims deletes managed claims that no longer back any worker.
//
// A claim deliberately outlives the worker that cut it, so the next worker of
// the same shape reuses a warm allocation instead of waiting for DRA again.
// That is also why claims carry no owner reference: a shared claim belongs to
// no single worker, and garbage-collecting it with its creator would pull the
// allocation out from under everyone else still on it.
//
// Four stays of execution: an OpenRLWorker names it, a live pod sits on it,
// DRA still reserves it, or it is too young to judge.
func (r *OpenRLWorkerReconciler) reclaimIdleClaims(ctx context.Context) error {
	logger := log.FromContext(ctx)

	var claims resourcev1.ResourceClaimList
	if err := r.List(ctx, &claims, client.InNamespace(r.Namespace), client.MatchingLabels{LabelManaged: "true"}); err != nil {
		return err
	}
	if len(claims.Items) == 0 {
		return nil
	}

	spokenFor := map[string]bool{}

	var workers openrlv1alpha1.OpenRLWorkerList
	if err := r.fleetReader().List(ctx, &workers, client.InNamespace(r.Namespace)); err != nil {
		return err
	}
	for i := range workers.Items {
		if name := workers.Items[i].Status.ClaimName; name != "" {
			spokenFor[name] = true
		}
	}

	var pods corev1.PodList
	if err := r.List(ctx, &pods, client.InNamespace(r.Namespace), client.HasLabels{LabelClaim}); err != nil {
		return err
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Status.Phase == corev1.PodSucceeded || pod.Status.Phase == corev1.PodFailed {
			continue
		}
		spokenFor[pod.Labels[LabelClaim]] = true
	}

	for i := range claims.Items {
		claim := &claims.Items[i]
		switch {
		case spokenFor[claim.Name]:
			continue
		case len(claim.Status.ReservedFor) > 0:
			continue
		case time.Since(claim.CreationTimestamp.Time) < claimGracePeriod:
			continue
		}
		logger.Info("reclaiming idle claim", "claim", claim.Name)
		if err := r.Delete(ctx, claim); err != nil && !apierrors.IsNotFound(err) {
			logger.Error(err, "failed to delete idle claim", "claim", claim.Name)
		}
	}
	return nil
}
