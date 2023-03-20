import os
import openai
import json
import numpy as np
from numpy.linalg import norm
import re
from time import time,sleep
from uuid import uuid4
import datetime

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_file(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        outfile.write(content)


def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return json.load(infile)


def save_json(filepath, payload):
    with open(filepath, 'w', encoding='utf-8') as outfile:
        json.dump(payload, outfile, ensure_ascii=False, sort_keys=True, indent=2)


def timestamp_to_datetime(unix_time):
    return datetime.datetime.fromtimestamp(unix_time).strftime("%A, %B %d, %Y at %I:%M%p %Z")


def gpt3_embedding(content, engine='text-embedding-ada-002'):
    max_retry = 5
    retry = 0

    while True:
        try:
            content = content.encode(encoding='ASCII',errors='ignore').decode()
            print("EMBEDDING SENT")
            response = openai.Embedding.create(input=content,engine=engine)
            print("EMBEDDING RECEIVED")
            vector = response['data'][0]['embedding']
            return vector

        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)

def similarity(v1, v2):
    # based upon https://stackoverflow.com/questions/18424228/cosine-similarity-between-2-number-lists
    return np.dot(v1, v2)/(norm(v1)*norm(v2))  # return cosine similarity

def fetch_memories(vector, logs, count):
    def is_message_in_prev_list(scores, message):
        for score in scores:
            if score['prev'] == message:
                return True
        return False

    scores = list()
    prev = None

    for i in logs:
        if vector == i['vector']:
            # skip this one because it is the same message
            continue
        score = similarity(i['vector'], vector)
        i['score'] = score
        i['prev'] = prev

        if not is_message_in_prev_list(scores, i):
            scores.append(i)

        prev = i
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    # TODO - pick more memories temporally nearby the top most relevant memories
    try:
        ordered = ordered[0:count]
        return ordered
    except:
        return ordered


def load_convo():
    files = os.listdir('nexus')
    files = [i for i in files if '.json' in i]  # filter out any non-JSON files
    result = list()
    for file in files:
        data = load_json('nexus/%s' % file)
        result.append(data)
    ordered = sorted(result, key=lambda d: d['time'], reverse=False)  # sort them all chronologically
    return ordered


def summarize_memories(memories):  # summarize a block of memories into one payload
    memories = sorted(memories, key=lambda d: d['time'], reverse=False)  # sort them chronologically
    block = ''
    blockInput = ''
    identifiers = list()
    timestamps = list()
    for mem in memories:
        block += mem['message'] + '\n\n'
        blockInput += mem['speaker'] + ': ' + mem['message'] + '\n\n'
        
        identifiers.append(mem['uuid'])
        timestamps.append(mem['time'])
    block = block.strip()
    prompt = open_file('prompts/prompt_notes.txt').replace('<<INPUT>>', blockInput)
    # TODO - do this in the background over time to handle huge amounts of memories
    notes = gpt3_completion(prompt)
    ####   SAVE NOTES
    vector = gpt3_embedding(block)
    info = {'notes': notes, 'uuids': identifiers, 'times': timestamps, 'uuid': str(uuid4()), 'vector': vector, 'time': time()}
    filename = 'notes_%s.json' % time()
    save_json('internal_notes/%s' % filename, info)
    return notes


def get_last_messages(conversation, limit):
    try:
        short = conversation[-limit:]
    except:
        short = conversation
    output = ''
    for i in short:
        output += '%s: %s\n\n' % (i['speaker'], i['message'])
    output = output.strip()
    return output

def gpt3_completion(inputText, engine='text-curie-001'):
    max_retry = 5
    retry = 0    
    
    text_tokens = round(len(inputText)/4.0)
    
    if (engine=='text-curie-001'):
        tokens = 2000 - text_tokens
    else:
        tokens = 4048 - text_tokens
    
    while True:
    
        try:

            print("API SENT")
            response = openai.Completion.create(
                engine=engine,
                prompt=inputText,
                temperature=0.5,
                max_tokens=tokens,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=['USER:', 'MAVEN:']
            )
            
            print("API RECEIVED")
            
            text = response['choices'][0]['text'].strip()
            text = re.sub('[\r\n]+', '\n', text)
            text = re.sub('[\t ]+', ' ', text)
            
            filename = '%s_gpt3.txt' % time()
            
            if not os.path.exists('gpt3_logs'):
                os.makedirs('gpt3_logs')
                
            save_file('gpt3_logs/%s' % filename, inputText + '\n\n==========\n\n' + text)
            
            return text
            
        except Exception as oops:

            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


if __name__ == '__main__':
    openai.api_key = open_file('openaiapikey.txt')
    
    
    conversation = load_convo()
    recent = get_last_messages(conversation, 1)
     
    print("RECENTLY\n")
    print(recent)
     
    while True:
        #### get user input, save it, vectorize it, etc
        userInput = input('\n\nUSER: ')

        block = ''

        if len(userInput.strip()) > 0:
            timestamp = time()
            vector = gpt3_embedding(userInput)
            timestring = timestamp_to_datetime(timestamp)

            # Save User's Message to Nexus
            info = {'speaker': 'USER', 'time': timestamp, 'vector': vector, 'message': userInput, 'uuid': str(uuid4()), 'timestring': timestring}
            filename = 'log_%s_USER.json' % timestamp
            save_json('nexus/%s' % filename, info)
        
            #### load conversation
            conversation = load_convo()
            #### compose corpus (fetch memories, etc)
            memories = fetch_memories(vector, conversation, 7)  # pull episodic memories

            for mem in memories:
                prev = mem['prev']
                if prev != None:
                    block += prev['speaker'] + ': ' + prev['message'] + '\n\n'

                block += mem['speaker'] + ': ' + mem['message'] + '\n\n----\n\n'


        # TODO - fetch declarative memories (facts, wikis, KB, company data, internet, etc)
      #  notes = summarize_memories(memories)
        # TODO - search existing notes first
        recent = get_last_messages(conversation, 20)
        prompt = open_file('prompts\prompt_response.txt').replace('<<NOTES>>', block).replace('<<CONVERSATION>>', recent)
        #### generate response, vectorize, save, etc
        output = gpt3_completion(prompt, 'text-davinci-003')
        #output = gpt3_completion(prompt)
        timestamp = time()
        vector = gpt3_embedding(output)
        print('\n\nMAVEN: %s' % output) 
        
        timestring = timestamp_to_datetime(timestamp)
        message = '%s:\n%s - %s' % ('MAVEN', timestring, output)
        info = {'speaker': 'MAVEN', 'time': timestamp, 'vector': vector, 'message': output, 'uuid': str(uuid4()), 'timestring': timestring}
        filename = 'log_%s_MAVEN.json' % time()
        save_json('nexus/%s' % filename, info)
        #### print output