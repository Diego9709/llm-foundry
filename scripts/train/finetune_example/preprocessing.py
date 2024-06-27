# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

r"""Example custom preprocessing function.

This is here to help illustrate the way to set up finetuning
on a local dataset. One step of that process is to create
a preprocessing function for your dataset, and that is what
is done below. Check out the LLM Finetuning section of
`../README.md` for more context.

For this example, we're going to pretend that our local dataset
is `./train.jsonl`.

Note: this dataset is actually a copy of one of our ARC-Easy
multiple-choice ICL eval datasets. And you would never actually
train on eval data! ... But this is just a demonstration.

Every example within the dataset has the format:
{
    'query': <query text>,
    'choices': [<choice 0 text>, <choice 1 text>, ...],
    'gold': <int> # index of correct choice
}

To enable finetuning, we want to turn this into a prompt/response
format. We'll structure prompts and responses like this:
{
    'prompt': <query text>\nOptions:\n - <choice 0 text>\n - <choice 1 text>\nAnswer: ,
    'response': <correct choice text>
}
"""

from typing import Dict, List, Union


def multiple_choice(
        inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
    PROMPT_FORMAT = '{query}\nOptions:{options}\nAnswer: '
    options = ''
    assert isinstance(inp['choices'], List)
    for option in inp['choices']:
        options += f'\n - {option}'
    query = inp['query']

    assert isinstance(inp['gold'], int)
    return {
        'prompt': PROMPT_FORMAT.format(query=query, options=options),
        'response': inp['choices'][inp['gold']],
    }


def normal_qa(inp):
    PROMPT_FORMAT = '### Instruction:\t\n{instruction}\n\n' 
    PROMPT_FORMAT2 = '### Instruction:\t\n{instruction}\n{input}\n\n'
    OUTPUT_FORMAT = '### Output:\t\n{output} </s>'

    if 'input' in inp.keys() and inp['input'] != '':
        return {
            'prompt': PROMPT_FORMAT2.format(instruction=inp['instruction'], input=inp['input']),
            'response': OUTPUT_FORMAT.format(output=inp['output'])
        }
    else:
        return {
            'prompt': PROMPT_FORMAT.format(instruction=inp['instruction']),
            'response': OUTPUT_FORMAT.format(output=inp['output'])
        }             
        
        
def nothing(inp):
    return {
        'prompt': inp['input'],
        'response': inp['output']
    }       
    
    
def chat(inp):
    
    user_template =  '### Instruction:\t\n{text}\n\n'  
    assistant_template = '### Output:\t\n{text} </s>'
    
    prompt = ''
    for i in inp['conversations'][:-1]:
        if i['from']=='user':
            conv = user_template.format(text=i['value'])
        if i['from']=='assistant':
            conv = assistant_template.format(text=i['value'])
        prompt += conv
        
    response = assistant_template.format(text=inp['conversations'][-1]['value'])
    
    return {
        'prompt': prompt,
        'response': response
    }        