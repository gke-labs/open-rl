package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// WorkerRole is which half of the training loop a worker runs. It selects
// which node pools may host the worker, and nothing else.
// +kubebuilder:validation:Enum=trainer;sampler
type WorkerRole string

const (
	RoleTrainer WorkerRole = "trainer"
	RoleSampler WorkerRole = "sampler"
)

// Phase is a coarse summary of where a worker is in scheduling.
// +kubebuilder:validation:Enum=Pending;Placing;Running;Failed
type Phase string

const (
	// PhasePending means no claim has been assigned, usually for want of capacity.
	PhasePending Phase = "Pending"
	// PhasePlacing means a claim and pod exist but the allocation is not yet observed.
	PhasePlacing Phase = "Placing"
	// PhaseRunning means the pod is running on an observed allocation.
	PhaseRunning Phase = "Running"
	// PhaseFailed means the request cannot be satisfied as written.
	PhaseFailed Phase = "Failed"
)

// Condition types set on an OpenRLWorker.
const (
	// ConditionPlaced reports whether the worker holds a claim and a pod.
	ConditionPlaced = "Placed"
)

// PodTemplateRef names the ConfigMap holding the pod spec this worker is
// rendered from.
//
// The template carries only the things scheduling has no opinion about: image,
// volumes, probes, security context. The controller owns nodeSelector and
// spec.resourceClaims, because those are the decision. A template that pins an
// accelerator SKU or a ResourceClaim name defeats the point, so both are
// overwritten rather than merged.
type PodTemplateRef struct {
	// Name of the ConfigMap in the controller's namespace.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Key within the ConfigMap holding the pod YAML.
	// +kubebuilder:default=pod.yaml
	Key string `json:"key,omitempty"`
}

// ContainerOverlay is what the caller stamps onto the template's first
// container. It exists so the controller never has to know anything about the
// model: everything the pod needs arrives here.
type ContainerOverlay struct {
	// Image replaces the template's image when set.
	Image string `json:"image,omitempty"`

	// Command replaces the template's command when non-empty.
	Command []string `json:"command,omitempty"`

	// Args are appended to the template's args.
	Args []string `json:"args,omitempty"`

	// Env entries are merged by name, overwriting the template's values.
	Env []corev1.EnvVar `json:"env,omitempty"`
}

// OpenRLWorkerSpec is one worker process's scheduling request.
//
// Four fields decide everything: who it is, where it may run, what it may run
// beside, and how much memory it needs. Memory arrives already estimated; the
// controller decides how many devices to spread it over and which claim to put
// it on, and never re-estimates.
type OpenRLWorkerSpec struct {
	// Role selects which node pools may host this worker. A node opts in per
	// role via openrl.io/trainer and openrl.io/sampler; a node that wants both
	// simply carries both labels.
	//
	// Role does not partition claims. On a node that accepts both, a trainer
	// and a sampler can share one accelerator by taking turns.
	Role WorkerRole `json:"role"`

	// ModelID is the open-rl model id this worker serves. It also determines
	// the pod name, so a repeated request reuses the existing worker.
	// +kubebuilder:validation:MinLength=1
	ModelID string `json:"modelId"`

	// Cohort is the set of workers this one may hold the accelerators
	// alongside. Memory sums within a cohort; cohorts take turns.
	//
	// The controller never interprets the string, it only compares it, so the
	// caller decides what sharing means. Several lora adapters over one frozen
	// base model name that base model. A full fine-tune shares nothing, leaves
	// this empty, and takes its turn alone -- which is the safe default,
	// because taking a turn always works and summing memory with strangers
	// does not.
	Cohort string `json:"cohort,omitempty"`

	// Memory is the total accelerator memory this worker needs. When it exceeds
	// one device the model is laid out across as many as it takes, so this is an
	// aggregate and not a per-device figure.
	Memory resource.Quantity `json:"memory"`

	// PodTemplate names the ConfigMap the worker pod is rendered from. When
	// unset the controller falls back to its role-default template.
	PodTemplate *PodTemplateRef `json:"podTemplate,omitempty"`

	// Container is what the caller stamps onto the rendered container.
	Container *ContainerOverlay `json:"container,omitempty"`
}

// OpenRLWorkerStatus is what the controller decided and what came of it.
type OpenRLWorkerStatus struct {
	// Phase is a coarse summary; Conditions carry the detail.
	Phase Phase `json:"phase,omitempty"`

	// DeviceCount is how many accelerators the controller decided this worker
	// spans. A decision, not a request: derived from Memory and the capacity
	// the pools registered.
	DeviceCount int32 `json:"deviceCount,omitempty"`

	// MemoryPerDevice is what each of those devices must provide, i.e. Memory
	// divided by DeviceCount and rounded up.
	MemoryPerDevice string `json:"memoryPerDevice,omitempty"`

	// HostMemoryPerResident is the host RAM this worker occupies while parked.
	//
	// cuda-checkpoint moves a suspended process's device memory into that
	// process's own host address space, so a worker costs its full accelerator
	// footprint in host RAM for as long as its cohort is not the one running.
	// This is what actually bounds how many residents a node can carry, and it
	// is why openrl.io/max-residents is a ceiling rather than the whole story.
	HostMemoryPerResident string `json:"hostMemoryPerResident,omitempty"`

	// EstimatedSwitchTime is how long the timeslicer is expected to spend
	// parking this worker and restoring the next one, from the measured cost
	// model in internal/placement. Reported so an operator can see what a
	// shared claim costs before the slice quantum makes it obvious.
	EstimatedSwitchTime string `json:"estimatedSwitchTime,omitempty"`

	// ClaimName is the ResourceClaim this worker was assigned to.
	ClaimName string `json:"claimName,omitempty"`

	// PodName is the worker pod, once created.
	PodName string `json:"podName,omitempty"`

	// NodeName is set only once the claim's allocation is observed. Until then
	// the node is genuinely unknown.
	NodeName string `json:"nodeName,omitempty"`

	// Reason is the most recent human-readable explanation of Phase.
	Reason string `json:"reason,omitempty"`

	// ObservedGeneration is the spec generation this status was computed from.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// Conditions holds the Placed condition and its history.
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=orw
// +kubebuilder:printcolumn:name="Role",type=string,JSONPath=`.spec.role`
// +kubebuilder:printcolumn:name="Cohort",type=string,JSONPath=`.spec.cohort`
// +kubebuilder:printcolumn:name="GPUs",type=integer,JSONPath=`.status.deviceCount`
// +kubebuilder:printcolumn:name="MemEach",type=string,JSONPath=`.status.memoryPerDevice`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Claim",type=string,JSONPath=`.status.claimName`
// +kubebuilder:printcolumn:name="Node",type=string,JSONPath=`.status.nodeName`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// OpenRLWorker is the scheduling request for a single worker process.
//
// The caller does not create pods or ResourceClaims. It creates one
// OpenRLWorker per worker it wants, carrying the estimated memory and the
// cohort that governs sharing. The controller turns that into a ResourceClaim
// (matched or created) and a pod, and records what it picked in status.
//
// Splitting the request from the decision is what makes scheduling
// inspectable: `kubectl get openrlworkers` shows what was asked for, what it
// was placed on, and why a pending worker is still pending.
type OpenRLWorker struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   OpenRLWorkerSpec   `json:"spec"`
	Status OpenRLWorkerStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// OpenRLWorkerList is a list of OpenRLWorker.
type OpenRLWorkerList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []OpenRLWorker `json:"items"`
}

func init() {
	SchemeBuilder.Register(&OpenRLWorker{}, &OpenRLWorkerList{})
}
