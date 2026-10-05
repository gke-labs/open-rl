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
// +kubebuilder:validation:Enum=Pending;Placing;Running;Succeeded;Failed
type Phase string

const (
	// PhasePending means no claim has been assigned, usually for want of capacity.
	PhasePending Phase = "Pending"
	// PhasePlacing means a claim and pod exist but the allocation is not yet observed.
	PhasePlacing Phase = "Placing"
	// PhaseRunning means the pod is running on an observed allocation.
	PhaseRunning Phase = "Running"
	// PhaseSucceeded means the pod's process exited cleanly. The seat and
	// claim stay held until the owner deletes the workload.
	PhaseSucceeded Phase = "Succeeded"
	// PhaseFailed means the request cannot be satisfied as written, or the
	// pod's process exited nonzero.
	PhaseFailed Phase = "Failed"
)

// Condition types set on an Workload.
const (
	// ConditionPlaced reports whether the worker holds a claim and a pod.
	ConditionPlaced = "Placed"
)

// TrainingKind is how the workload trains. Informational; placement reads
// Exclusive.
// +kubebuilder:validation:Enum=fft;lora
type TrainingKind string

const (
	TrainingKindFFT  TrainingKind = "fft"
	TrainingKindLoRA TrainingKind = "lora"
)

// AcceleratorMode names the claim shape the workload is asking for.
// SingleGPU is one device with room for Memory. MultiGPU is Devices whole
// devices on one node for a runtime that drives them as one torchrun group
// (tensor, context or data parallel); such a claim is exclusive and never
// time-slice shared.
// +kubebuilder:validation:Enum=SingleGPU;MultiGPU
type AcceleratorMode string

const (
	AcceleratorModeSingleGPU AcceleratorMode = "SingleGPU"
	AcceleratorModeMultiGPU  AcceleratorMode = "MultiGPU"
)

// AcceleratorSpec is the estimator's accelerator requirement: a claim shape
// the process can actually drive, so placement never guesses a count.
// +kubebuilder:validation:XValidation:rule="self.mode != 'MultiGPU' || (has(self.devices) && self.devices >= 2)",message="MultiGPU needs devices >= 2"
// +kubebuilder:validation:XValidation:rule="self.mode == 'MultiGPU' || !has(self.devices)",message="devices is only for MultiGPU"
type AcceleratorSpec struct {
	// Mode is the claim shape Memory describes.
	// +kubebuilder:default=SingleGPU
	Mode AcceleratorMode `json:"mode,omitempty"`

	// Memory is peak accelerator memory: the total for SingleGPU, the peak
	// per device for MultiGPU. Never re-estimated.
	Memory resource.Quantity `json:"memory"`

	// Devices is how many devices a MultiGPU claim asks for, all on one node.
	// +kubebuilder:validation:Minimum=2
	// +optional
	Devices int32 `json:"devices,omitempty"`
}

// WorkloadSpec is one worker process's scheduling request. Memory arrives
// already estimated; the controller decides how many devices to spread it
// over and which claim to put it on, and never re-estimates.
type WorkloadSpec struct {
	// Role selects which node pools may host this worker, via the
	// openrl.io/trainer and openrl.io/sampler labels. It does not partition
	// claims: on a node that accepts both, the two roles share by turns.
	Role WorkerRole `json:"role"`

	// ModelID names the model this worker serves: configuration for the
	// worker process and the humans reading kubectl get, never identity.
	// The worker's identity is metadata.name.
	// +kubebuilder:validation:MinLength=1
	ModelID string `json:"modelID"`

	// OwnerID is the unit of fairness: the runtime serves owners round-robin,
	// so an owner never gets extra turns for having more processes, requests,
	// or adapters. Opaque to the controller, never read by placement -- one
	// worker is resident at a time whatever the owners. A worker naming no
	// owner is an owner of one.
	OwnerID string `json:"ownerID,omitempty"`

	// TrainingKind records whether this runtime is full fine-tuning or LoRA.
	// Placement does not read it.
	TrainingKind TrainingKind `json:"trainingKind,omitempty"`

	// Exclusive keeps this worker alone on its GPU. Two workers share a
	// claim only when neither is exclusive. The API server sets it when the
	// worker cannot suspend between turns, as LoRA workers cannot, or when
	// the model asked to share nothing. Omitted means exclusive.
	// +kubebuilder:default=true
	// +optional
	Exclusive bool `json:"exclusive"`

	// Accelerator is the estimator's requirement this workload is placed by.
	Accelerator AcceleratorSpec `json:"accelerator"`

	// WorkerContainerName names the container in Template that consumes the
	// accelerator: the claim reference and the time-slice group land there.
	// +kubebuilder:default=worker
	WorkerContainerName string `json:"workerContainerName,omitempty"`

	// Template is the complete worker pod. The API server supplies every
	// field knowable before placement -- image, command, identity env,
	// resources, volumes, tolerations. The controller stamps only what
	// placement decided: the claim reference, eligible-node affinity, the
	// time-slice group, and the owner reference. Placement-owned fields in
	// the template (nodeName, node selectors, required node affinity, claim
	// references, accelerator extended resources, the time-slice group) are
	// rejected rather than merged.
	// +kubebuilder:validation:XValidation:rule="!has(self.spec.nodeName)",message="nodeName is placement's decision; leave it unset"
	// +kubebuilder:validation:XValidation:rule="!has(self.spec.nodeSelector)",message="nodeSelector is placement's decision; leave it unset"
	// +kubebuilder:validation:XValidation:rule="!has(self.spec.affinity)",message="affinity is placement's decision; leave it unset"
	// +kubebuilder:validation:XValidation:rule="!has(self.spec.resourceClaims)",message="resource claims are placement's decision; leave them unset"
	Template corev1.PodTemplateSpec `json:"template"`
}

// WorkloadStatus is what the controller decided and what came of it.
type WorkloadStatus struct {
	// Phase is a coarse summary; Conditions carry the detail.
	Phase Phase `json:"phase,omitempty"`

	// DeviceCount is how many accelerators the controller decided this worker
	// spans. A decision, not a request: derived from Memory and the capacity
	// the pools registered.
	DeviceCount int32 `json:"deviceCount,omitempty"`

	// MemoryPerDevice is what each of those devices must provide, i.e. Memory
	// divided by DeviceCount and rounded up.
	MemoryPerDevice string `json:"memoryPerDevice,omitempty"`

	// ClaimName is the ResourceClaim this worker was assigned to. The seat
	// itself lives on the claim's ClaimLedger, whose name is derived
	// from the claim's.
	ClaimName string `json:"claimName,omitempty"`

	// AssignmentID is this worker's current booking in that ledger. It must
	// match the ledger's seat and the pod's stamp; a mismatch marks a stale
	// pod or a superseded assignment.
	AssignmentID string `json:"assignmentID,omitempty"`

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
// +kubebuilder:resource:shortName=owl
// +kubebuilder:printcolumn:name="Role",type=string,JSONPath=`.spec.role`
// +kubebuilder:printcolumn:name="Owner",type=string,JSONPath=`.spec.ownerID`
// +kubebuilder:printcolumn:name="GPUs",type=integer,JSONPath=`.status.deviceCount`
// +kubebuilder:printcolumn:name="MemEach",type=string,JSONPath=`.status.memoryPerDevice`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Claim",type=string,JSONPath=`.status.claimName`
// +kubebuilder:printcolumn:name="Node",type=string,JSONPath=`.status.nodeName`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// Workload is the scheduling request for a single worker process. The
// caller creates one per worker it wants; the controller turns it into a
// ResourceClaim (matched or created) and a pod, and records what it picked in
// status -- so `kubectl get workloads` shows what was asked for, where it
// went, and why a pending worker is still pending.
type Workload struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// The spec is immutable: every field either places the worker or renders
	// its pod, and V1 does not re-place or re-render a live worker. Change by
	// deleting and recreating.
	// +kubebuilder:validation:XValidation:rule="self.role == oldSelf.role && self.modelID == oldSelf.modelID && has(self.ownerID) == has(oldSelf.ownerID) && (!has(self.ownerID) || self.ownerID == oldSelf.ownerID) && has(self.trainingKind) == has(oldSelf.trainingKind) && (!has(self.trainingKind) || self.trainingKind == oldSelf.trainingKind) && self.exclusive == oldSelf.exclusive && self.accelerator == oldSelf.accelerator",message="placement fields are immutable; delete and recreate the workload"
	Spec   WorkloadSpec   `json:"spec"`
	Status WorkloadStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// WorkloadList is a list of Workload.
type WorkloadList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []Workload `json:"items"`
}

func init() {
	SchemeBuilder.Register(&Workload{}, &WorkloadList{})
}
