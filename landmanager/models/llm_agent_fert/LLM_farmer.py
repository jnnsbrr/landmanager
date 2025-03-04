#get needed packages
try:
  !pip -q install openai cohere tiktoken
  import pandas as pd
  import numpy as np
  import openai
  import time
  import random
  from matplotlib import pyplot as plt
  print("Python modules installed successfully!")
except:
  print("Error! One or more libraries could not be installed! Contact us.")


#define the world
class world():
    #need adjustment so that it waorks with copan_LPJmL
    #init farmers with their needed attributes and knowlege about past fertilizer use
  xxx

#get LLM involved
client = openai.OpenAI(api_key="your-api-key-here")

class farmer():

    def __init__(self, unique_id, model, name, traits, clothes = None):
        self.cell = xx
        self.country = xx
        self.size = xx
        self.crops = np.array([],dtype=int)
        self.past_knowlege = np.array([],dtype=int)
  


    def get_output_from_chatgpt(self, messages,
                                model ="gpt-3.5-turbo-0613",temperature =0): #adjust LLM model used
        success = False
        retry = 0
        max_retries = 30
        while retry < max_retries and not success:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,  # this is the degree of randomness of the model's output
                    )

                success = True
            except Exception as e :
                print(f"Error: { e }\ nRetrying ...")
                retry +=1
                time.sleep(0.5)

        return response.choices[0].message.content


    def decide_fertilizer(self) : #finde raus wie wir hier gut auf eine Tabelle zugreifen können
        question_prompt = f"""
        You are a farmer on in {country} with a from of size {self.model.farm_size} hectares. On this farm you are growing different crops in different amounts. Those crops are:
        {crop 1} with a share of {crop 1 share} ,
        {crop 2} with a share of {crop 2 share} ,
        {crop 3} with a share of {crop 3 share} ,
        {crop 4} with a share of {crop 4 share} ,
        {crop 5} with a share of {crop 5 share} ,
        {crop 6} with a share of {crop 6 share} ,
        {crop 7} with a share of {crop 7 share} ,
        {crop 8} with a share of {crop 8 share} ,
        {crop 9} with a share of {crop 9 share} .
        Now, You have to decide how much fertilizer you want to use for each crop type on your farm. Your aim is to increase you crop yield by staing withing the planetary boundaries.
        In  the past years the amout of applied fertiliser per crop and the resulting years were the following:
        Last year you apllied {fertilizer use last year} fertilizer to {crop 1} and the crop yield was {crop yield last year}
        2 year ago you apllied {fertilizer use 2 years ago} fertilizer to {crop 1} and the crop yield was {crop yield 2 years ago}
        3 year ago you apllied {fertilizer use 3 years ago} fertilizer to {crop 1} and the crop yield was {crop yield 3 years ago}
        ...
        Based on the above context, you need to decide how much ferlizer you want to apply this year to each of your crops.
        You must provide your choices by naming the crop followed by an integer number representing the amount of fertilizer you want to apply to that crop. 
        Plase use a new line to separate different crop names and the respective amounts of fertilizer.
        For example, if you want to aplly an fertilizer amount of 4 to maize and 2 to corn, your response will be:
        Response: 
        [maize, 4
        corn, 2]
        Please make sure, you answer in this exact structure.
        """

        messages = [{'role':'system', 'content':question_prompt}]
        try:
            output = self.get_output_from_chatgpt(messages)
        except Exception as e:
            print(f"{e}\nProgram paused. Retrying after 60s...")
            time.sleep(60)
            output = self.get_output_from_chatgpt(messages)
        reasoning = ""
        response = ""
        try:
            intermediate = output.split("Reasoning:",1)[1]
            reasoning , response = intermediate.split ("Response:")
            response = response.strip().split("." ,1)[0]
            reasoning = reasoning.strip()

        except:
            print("Reasoning or response were not parsed correctly.")
            print(f"Output: {output}")
            response = "0"
            reasoning = "N/A"
        reasoning = reasoning.strip()
        response = response.strip()
        print(f"{self.name}’s reasoning: {reasoning}")
        print(f"{self.name}’s response: {response}")
        if response.lower() == "blue": #needs adjustment - add response table with fertilizer use and delete 11th row
            self.clothes = "blue"
            self.mem = np.append(self.mem,1)
        elif response.lower() == "green":
            self.clothes = "green"
            self.mem = np.append(self.mem, 0)
        else:
            print("Warning! Response was unclear. Reading as if no fertilizer has been applied.")
            print(f"Response was: {response}")
            self.mem = np.append(self.mem, 1)




