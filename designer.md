## Raw Thoughts

The purpose of this project is to build an API for post training LLM models.

About the API specification,

we are inspired by the beautiful API provided by Tinker, training API providered by Thinking Machines.

So we don't mind exactly implementing tinker compatible APIs so that we can use tinker client to interact with our API.

Tinker end user docs are located at https://tinker-docs.thinkingmachines.ai/

Under the hood page gives a little idea about the architecture of Tinker. https://tinker-docs.thinkingmachines.ai/under-the-hood 

End user docs are located in llms-full.txt in this directory or https://tinker-docs.thinkingmachines.ai/llms-full.txt


## Dev environment

Since training API implementation requires GPUs so I will be using a remote linux machine (ubuntu) for running the API.

However, I will be doing most of my development locally on my MAC.

However, I will like to test the client of the API locally on my MAC.

And we will use `uv` to manage the dependencies.

I think it will make sense to have `server` and `client` directories with each having its own `pyproject.toml` file.

I would like to just rsync the `server` directory to the remote machine and run it there.

And I would like to just rsync the `client` directory to my MAC and run it there.


## Build in Stages

I would like to first get a basic MVP working with assuming API has just single user and single model.

I would also like to just use a simple use-case of SFT working and then extend it to RL use-cases.


## OSS libraries

Post training will involve using training as well inference components under the hood and we would like to use OSS libraries for that.
Think we want to assemble existing OSS libraries to build our API.

For inference, we can use vllm

Huggingface's TRL library is also promising.


