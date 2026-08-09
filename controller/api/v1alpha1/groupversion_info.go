// Package v1alpha1 contains the OpenRLWorker API, the placement request the
// gateway writes for every worker process it wants running.
//
// See docs/designs/012-dynamic-placement.md.
// +kubebuilder:object:generate=true
// +groupName=openrl.io
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	// GroupVersion is the group and version this package's types belong to.
	GroupVersion = schema.GroupVersion{Group: "openrl.io", Version: "v1alpha1"}

	// SchemeBuilder registers this package's types with a runtime.Scheme.
	SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}

	// AddToScheme adds this package's types to a runtime.Scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)
