# fio from fio-l4-lvw8

file: file: /mnt/shared/open-rl/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218/model-00001-of-00005.safetensors

| job | MiB/s | IOPS | mean latency ms | read | seconds |
|---|---|---|---|---|---|
| fio-seqread-1M-direct | 478.7 | 479 | 2.09 | 3.72 GiB | 8.0 |
| fio-randread-4k-direct | 4.2 | 1066 | 0.94 | 0.12 GiB | 30.0 |
| fio-seqread-256k-qd1-direct | 210.2 | 841 | 1.19 | 6.16 GiB | 30.0 |
| fio-seqread-1M-4jobs-direct | 1775.4 | 1775 | 2.25 | 52.02 GiB | 30.0 |
| fio-seqread-1M-buffered | 893.3 | 893 | 1.12 | 3.72 GiB | 4.3 |
