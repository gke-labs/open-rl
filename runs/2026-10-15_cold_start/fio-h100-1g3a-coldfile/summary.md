# fio from fio-h100-1g3a-coldfile

file: file: /mnt/shared/open-rl/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554/model-00001-of-00003.safetensors

| job | MiB/s | IOPS | mean latency ms | read | seconds |
|---|---|---|---|---|---|
| fio-seqread-1M-direct | 16.2 | 16 | 61.72 | 0.95 GiB | 60.2 |
| fio-randread-4k-direct | 0.1 | 33 | 30.18 | 0.00 GiB | 30.0 |
| fio-seqread-256k-qd1-direct | 44.8 | 179 | 5.58 | 1.31 GiB | 30.0 |
| fio-seqread-1M-4jobs-direct | 134.7 | 135 | 29.66 | 3.96 GiB | 30.1 |
| fio-seqread-1M-buffered | 60.4 | 60 | 16.55 | 3.54 GiB | 60.1 |
