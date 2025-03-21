from typing import cast
from matplotlib.pylab import Enum
from simplellm.llama import LLamaFirstStage, LLamaLastStage, LLamaStage # get our models
from simplellm.tokenizers import SPTokenizer # get our tokenizer
from simplellm.dataloaders import TinyStories # get our dataset
from simplellm.losses import causalLLMLoss # our loss
from torch.optim import SGD, Adam
import torch.nn.functional as F
import torch
import torch.distributed as dist
import os
from sys import argv


class Questions(Enum):
    """
    Enum for the homework part B questions
    """
    B1 = "B1"
    B2 = "B2"

# get/setup global args
rank = int(argv[1])
question = argv[2]


torch.manual_seed(0) # important so that every node compute the same thing and converge
dmodel = 288
num_heads = 6
seq_l = 256
batch_size = 3
device = "mps"
steps = 5000

class Microbatch:
    """
    A microbatch is a part of the data that is sent to the next stage of the pipeline.
    This class takes care of all the training algorithm regarding the microbatch.
    """
    def __init__(self, global_rank: int, net: LLamaStage, microbatch: torch.Tensor, group: dist.ProcessGroup):
        """
        Args:
            rank (int): the rank of the process
            net (LLamaStage): the model
            microbatch (torch.Tensor): the data to be sent to the next stage
        """
        self.net = net
        self.microbatch = microbatch
        self.microbatch_out: torch.Tensor = torch.empty((microbatch.size(0), microbatch.size(1), dmodel))
        self.global_rank = global_rank
        self.sent_work: dist.Work = None
        self.group = group
        
        self.rank = dist.get_group_rank(group, rank)


    def forward(self):
        print("sent microbatch")
        pass

    def backward(self):
        print("received gradients")
        pass

    def onDestroy(self):
        """""
        This method is called when the microbatch is no longer needed. 
        It should be used to free memory.
        """
        del self.microbatch_out
        del self.microbatch
        del self.sent_work

    def wait_sent(self):
        """
        This method waits all the sent works to be done.
        """
        if self.sent_work is not None:
            self.sent_work.wait()
            self.sent_work = None
        else:
            print("WARNING: no work to wait")


class MicrobatchHead(Microbatch):
    """
    This class represents the handling of a microbatch in the first stage of the pipeline.
    It embeds the microbatch's data and send it to the next stage without blocking.
    Afterwards and storing the activations, it waits for the gradients to be sent back and compute its own network's gradients.
    """
    def __init__(self, rank, net: LLamaFirstStage, microbatch: torch.Tensor, group: dist.ProcessGroup):
        super().__init__(rank, net, microbatch, group)
        # self.net = cast(LLamaFirstStage, net)

    def forward(self):
        self.microbatch_out: torch.Tensor = self.net.embed(self.microbatch)
        self.sent_work = dist.isend(self.microbatch_out.to("cpu"), self.global_rank+1, self.group)


    def backward(self):
        inp_grad = torch.empty((self.microbatch_out.size(0), self.microbatch_out.size(1), dmodel))
        work = dist.irecv(inp_grad, self.global_rank+1, self.group)
        work.wait()
        self.microbatch_out.backward(inp_grad.to(device))

class MicrobatchMiddle(Microbatch):
    """
    This class represents the handling of a microbatch in the middle stage of the pipeline. 
    It receives the microbatch's activations from the previous stage, computes the activations and sends them to the next stage without blocking.
    Afterwards, it waits for the gradients to be sent back and computes its own network's gradients and sends the gradients to the previous stage.
    """
    def __init__(self, global_rank: int, net: LLamaStage, microbatch: torch.Tensor, group: dist.ProcessGroup):
        super().__init__(global_rank, net, microbatch, group)

    def forward(self):
        previous_l_out = torch.empty((self.microbatch.size(0), self.microbatch.size(1), dmodel))
        work = dist.irecv(previous_l_out, self.global_rank-1, self.group)
        work.wait()

        with torch.no_grad():
            previous_l_out = previous_l_out.to(device)
            previous_l_out.requires_grad_()
            previous_l_out.retain_grad()

        self.microbatch_out = self.net(previous_l_out)
        self.prev_l_out = previous_l_out
        self.sent_work = dist.isend(self.microbatch_out.to("cpu"), self.global_rank+1, self.group)

    def backward(self):
        grad_l_next = torch.empty((self.microbatch_out.size(0), self.microbatch_out.size(1), dmodel))
        work = dist.irecv(grad_l_next, self.global_rank+1, self.group)
        work.wait()
        self.microbatch_out.backward(grad_l_next.to(device))
        self.sent_work = dist.isend(self.prev_l_out.grad.to("cpu"), self.global_rank-1, self.group)

class MicrobatchTail(Microbatch):
    """"
    This class represents the handling of a microbatch in the last stage of the pipeline.
    It receives the target from the previous stage, computes the activations and provides the code to compute the microbatch's loss, compute its gradients w.r.t. 
    the loss and send the gradients to the previous stage.
    """
    def __init__(self, global_rank:int, net: LLamaLastStage, microbatch: torch.Tensor, group: dist.ProcessGroup):
        super().__init__(global_rank, net, microbatch, group)

    def forward(self):
        prev_l_out = torch.empty((self.microbatch.size(0), self.microbatch.size(1), dmodel))
        work = dist.irecv(prev_l_out, self.global_rank-1, self.group)
        work.wait()
        with torch.no_grad():
            prev_l_out = prev_l_out.to(device)
            prev_l_out.requires_grad_()
            prev_l_out.retain_grad()
        self.microbatch_out = self.net(prev_l_out)
        self.prev_l_out = prev_l_out
    
    def compute_loss(self, tokenizer: SPTokenizer):
        target = self.microbatch
        logits = self.microbatch_out
        loss = causalLLMLoss(logits, target, tokenizer.vocab_size)
        self.loss = loss

    
    def backward(self):
        self.loss.backward()
        self.sent_work = dist.isend(self.prev_l_out.grad.to("cpu"), self.global_rank-1, self.group)


class PipelinePart:
    """"
    This class represents a part of the pipeline. It contains the logic to handle the forward and backward pass of the pipeline by minibatches.
    It also contains the logic to scale the gradients and clean up the memory after a minibatch is done
    """
    global_rank: int
    world_size: int
    ds: TinyStories
    net: LLamaFirstStage | LLamaLastStage | LLamaStage
    optim: torch.optim.Optimizer
    iter_ds: iter
    batch_size: int
    seq_l: int
    microbatches: list[Microbatch]
    group: dist.ProcessGroup

    def __init__(self, global_rank: int, group: dist.ProcessGroup, batch_size: int, seq_l: int):
        self.global_rank = global_rank
        self.world_size = group.size()
        self.batch_size = batch_size
        self.seq_l = seq_l
        # tokenizer = SPTokenizer()
        # self.ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l)
        # self.iter_ds = iter(self.ds)
        self.group = group
    
    def forward(self):
        pass

    def backward(self):
        pass

    def clean_up_memory(self):
        for microbatch in self.microbatches:
            microbatch.onDestroy()
            del microbatch
        del self.microbatches
        torch.cuda.empty_cache()
    
    def wait_all_sent(self):
        for microbatch in self.microbatches:
            microbatch.wait_sent()
    
    def scale_gradients(self):
        for param in self.net.parameters():
            if param.grad is not None:
                param.grad = param.grad / self.world_size


class PipelineHead(PipelinePart):
    """
    This class represents the beginning stage of a pipeline parallelism. 
    It handles the group the running node is in, the dataset, the model and the microbatches. 
    """
    def __init__(self, global_rank: int, group: dist.ProcessGroup, batch_size: int, seq_l: int):
        super().__init__(global_rank, group, batch_size, seq_l)
        
        print("global rank", global_rank)
        print("world size", self.world_size)

        parallel_rank = global_rank // self.world_size
        print("parallel rank", parallel_rank)
        tokenizer = SPTokenizer()
        self.ds = TinyStories(tokenizer,batch_size=batch_size, seq_l=seq_l, skip=steps*parallel_rank)
        self.iter_ds = iter(self.ds)

        global_size = dist.get_world_size()
        self.net = LLamaFirstStage(tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers= 6 // global_size, ctx_size=seq_l)
        

    def forward(self):
        """
        The forward pass run at the first stage in the pipeline. 
        It splits the minibatch into microbatches, processes them and waits until all the microbatches are sent to ensure the forward 
        pass is done in this stage.
        """
        out: torch.Tensor = next(self.iter_ds)
        out = out.to(device)
        self.microbatches = [MicrobatchHead(self.global_rank, self.net, chunk, self.group) for chunk in out.chunk(self.world_size)]
        for microbatch in self.microbatches:
            microbatch.forward()
        self.wait_all_sent()


    
    def backward(self):
        """
        The backward pass run at the first stage in the pipeline. It computes the gradients of the microbatches, cumulating them, and scales them.
        """
        for i, microbatch in enumerate(self.microbatches):
            microbatch.backward()
    
            # as improvement delete unused memory (activations, microbatch)
        self.scale_gradients()  

class PipelineMiddle(PipelinePart):
    """
    This class represents any middle stage of a pipeline parallelism.
    It handles the group the running node is in, the model and the microbatches.
    """
    def __init__(self, global_rank, group, batch_size, seq_l):
        super().__init__(global_rank, group, batch_size, seq_l)
        global_size = dist.get_world_size()
        self.net = LLamaStage(dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers=6 // global_size, ctx_size=seq_l)

    def forward(self):
        """
        The forward pass run at the middle stage in the pipeline.
        It creates the microbatches in charge of the microbatch handling and waits until all the microbatches are sent to ensure the forward pass is done in this stage.
        """
        stages = self.world_size
        microbatch_size = self.batch_size // stages
        dummy_microbatch = torch.empty((microbatch_size, self.seq_l, dmodel)).to(device)
        self.microbatches = [MicrobatchMiddle(self.global_rank, self.net, dummy_microbatch, self.group) for _ in range(stages)]
        for microbatch in self.microbatches:
            microbatch.forward()
        self.wait_all_sent()


    def backward(self):
        """
        The backward pass run at the middle stage in the pipeline. It computes the gradients of the microbatches, cumulating them, and scales them.
        """
        for microbatch in self.microbatches:
            microbatch.backward()
            # scale gradients
            # microbatch.microbatch_out.grad = microbatch.microbatch_out.grad / world_size
            # as improvement delete unused memory (activations, microbatch)
        self.scale_gradients()  
        self.wait_all_sent()

class PipelineTail(PipelinePart):
    """
    This class represents the last stage of a pipeline parallelism.
    It handles the group the running node is in, the dataset, the model and the microbatches.
    """

    def __init__(self, global_rank:int, group: dist.ProcessGroup, batch_size: int, seq_l: int):
        super().__init__(global_rank, group, batch_size, seq_l)
        print("global rank", global_rank)
        print("world size", self.world_size)

        parallel_rank = global_rank // self.world_size
        print("parallel rank", parallel_rank)
        self.tokenizer = SPTokenizer()
        self.ds = TinyStories(self.tokenizer,batch_size=batch_size, seq_l=seq_l, skip=steps*parallel_rank) # no skip
        self.iter_ds = iter(self.ds)

        global_size = dist.get_world_size()
        self.net = LLamaLastStage(self.tokenizer.vocab_size,dmodel=dmodel,num_heads=num_heads,
                    device=device, n_layers=6 // global_size, ctx_size=seq_l)

    def forward(self):
        """
        The forward pass run at the last stage in the pipeline.
        It splits the minibatch into microbatches and processes them.
        """
        target: torch.Tensor = next(self.iter_ds)
        target = target.to(device)
        self.microbatches = [MicrobatchTail(self.global_rank, self.net, chunk, self.group) for chunk in target.chunk(self.world_size)]
        self.microbatch_size = target.size(0) // self.world_size
        for microbatch in self.microbatches:
            microbatch.forward()
            microbatch.compute_loss(self.tokenizer)
        print(self.microbatches[-1].loss.item())

    
    def backward(self):
        """
        It computes all microbatches's gradients, cumulates them, scales them and sends them to the previous stage.
        """
        for microbatch in self.microbatches:
            microbatch.backward()
            # microbatch.microbatch_out.grad = microbatch.microbatch_out.grad / world_size
            # as improvement delete unused memory (activations, microbatch)
        self.scale_gradients()  
        self.wait_all_sent()


class ModelPipeline:
    """
    This class orchestrates the pipeline parallelism. It contains the logic to handle the pipeline stages and the communication between them.
    It also takes care of the group creating and nodes assignment into them.
    """
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self._init_pipeline_group()
        self._init_part(rank)
            
        self.optim = Adam(self.part.net.parameters(),lr=8e-4)

    def _init_pipeline_group(self):
        """
        Since we do not provide here also data parallelism, we just create the default group.
        """
        self.group: dist.ProcessGroup = dist.new_group()

    def _init_part(self, rank: int):
        """
        Initializes the part of the pipeline the node is in charge of.
        """
        dist.barrier()
        
        # we need to check the group's rank and not the global rank since the node is in a group and all the code runs 
        # in the context of the group
        group_rank = dist.get_group_rank(self.group, rank)
        group_world_size = dist.get_world_size(self.group)

        if group_rank == 0:
                self.part = PipelineHead(rank, self.group, batch_size, seq_l)
        elif group_rank == group_world_size - 1:
            self.part = PipelineTail(rank, self.group, batch_size, seq_l)
        else:
            self.part = PipelineMiddle(rank, self.group, batch_size, seq_l)
                
    def start(self):
        """
        Here is where the magic happens. This is the distributed training loop.
        First it resets the gradients, runs the forward pass, runs the backward pass and then updates the weights.
        Lastly, it cleans up the memory and empties the cache.
        """

        for _ in range(steps):
            self.optim.zero_grad()
            self.part.forward()
            self.part.backward()
            # this is kind of a barrier to make sure all gradients are delivered and computed
            self.optim.step()
            self.part.clean_up_memory()
            torch.cuda.empty_cache()

class DataParallelPipeline(ModelPipeline):
    """
    This class implements the ModelPipeline orchestrator but with data parallelism. 
    """

    def __init__(self, rank: int, world_size: int, parallel_pipelines: int):

        # check if the world size is divisible by the parallel pipelines in order to have the same number of nodes in each pipeline
        if world_size % parallel_pipelines != 0:
            raise ValueError("world_size must be divisible by parallel_pipelines")
        self.parallel_pipelines = parallel_pipelines
        self.parts_per_pipeline = world_size // parallel_pipelines
        super().__init__(rank, world_size)
        
        self._init_weight_sizes()

    def _init_pipeline_group(self):
        """
        Creates the groups for the pipeline parallelism. 
        In this case it's more complex since now we have to assign nodes to the given number of pipelines.
        """

        # create the horizontal groups. By horizontal is meant the nodes in the same pipeline, if you see a pipeline as an horizontal line.
        # here I splits the nodes into the number of parallel pipelines
        horizontal_group_ranks = torch.arange(0, self.world_size).chunk(self.parallel_pipelines)

        # convert the tensor into a list of lists
        horizontal_group_ranks = [group_ranks.tolist() for group_ranks in horizontal_group_ranks]

        # create the list groups, each group representing a pipeline
        self.groups = [dist.new_group(group_ranks) for group_ranks in horizontal_group_ranks]
        self.group = self.groups[self.rank // self.parts_per_pipeline]


    def _init_part(self, rank: int):
        super()._init_part(rank)

        # group_size = dist.get_world_size(self.group)
        # Here I also create vertical groups. 
        # The vertical groups are the groups that are in the same position in each pipeline.
        # For example, if we have 3 pipelines, the vertical groups are the groups of nodes with rank 0, 1, 2 in each pipeline.
        # This is needed to communicate the gradients between the pipelines for the same part of the pipeline, meaning the same part of the llm.
        pipelines_ranks = [dist.get_process_group_ranks(group) for group in self.groups]
        pipelines_ranks = torch.tensor(pipelines_ranks)
        vertical_groups:list[dist.ProcessGroup] = []
        
        for i in range(pipelines_ranks.size(1)):
            vertical_groups.append(dist.new_group(pipelines_ranks[:,i].tolist()))
        self.vertical_groups = vertical_groups
        self.vertical_group = self.vertical_groups[int(rank % self.parts_per_pipeline)]


    def _init_weight_sizes(self):
        # this is taken from the tutorial to aggregate the weights between the pipelines
        self.sizes:list[int] = []
        self.len_sizes:list[int] = []
        for param in self.part.net.parameters():
            self.sizes.append(param.shape)
            self.len_sizes.append(len(param.view(-1)))

    def start(self):
        for itr in range(steps):
            self.optim.zero_grad()
            self.part.forward()
            self.part.backward()

            # waits for all the gradients to be computed before aggregating all the pipelines gradients
            dist.barrier()
            self.aggregate_gradients()
            self.optim.step()
            self.part.clean_up_memory()
            torch.cuda.empty_cache()    
    
    def aggregate_gradients(self):

        # this code is also taken from the tutorial to aggregate the gradients between the pipelines. 
        # The only thing is new is the use of the vertical group.
        tmp = []
        for param in self.part.net.parameters():
            if param.grad == None:
                tmp.append(torch.zeros_like(param,device="cpu").view(-1))
                continue
            tmp.append(param.grad.view(-1).to("cpu"))
            param.grad = None
        prev_grad = torch.cat(tmp).to("cpu")
        dist.all_reduce(prev_grad, op = dist.ReduceOp.SUM, group=self.vertical_group)
        tmp = torch.split(prev_grad, self.len_sizes)
        for i, param in enumerate(self.part.net.parameters()):
            param.grad = tmp[i].view(self.sizes[i]).to(device) / self.parallel_pipelines # average


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    if question == Questions.B2.value:
        world_size = 6
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        
        # Do you see the same as I do? We can make any kind of joint architecture here.
        # I wanted to provide a shell code to show it as a playground but the bolierplate code is too much.
        model = DataParallelPipeline(rank, world_size, 2)
    else:
        world_size = 3
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        model = ModelPipeline(rank, 3)
    model.start()
    dist.destroy_process_group()