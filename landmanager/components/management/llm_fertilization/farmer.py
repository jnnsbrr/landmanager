"""Farmer entity type class of inseeds_farmer_management"""

# This file is part of pycopancore.
#
# Copyright (C) 2016-2017 by COPAN team at Potsdam Institute for Climate
# Impact Research
#
# URL: <http://www.pik-potsdam.de/copan/software>
# Contact: core@pik-potsdam.de
# License: BSD 2-clause license
import numpy as np
import time
import ast
import pandas as pd
import xarray as xr

import pycopancore.model_components.base as core
import landmanager.components.base as base
import landmanager.components.management as management


class Farmer(management.Farmer):
    """Farmer (Individual) entity type mixin class."""

    # standard methods:
    def __init__(self, **kwargs):
        """Initialize an instance of Farmer."""
        super().__init__(**kwargs)  # must be the first line

        # TODO: mask zeros in cftfrac
        self.name = f"Farmer {self.cell.grid.cell.item()}"
        self.position = (
            f"Lat: {self.cell.grid.lat.item()}, Lon: {self.cell.grid.lon.item()}"  # noqa
        )

        mask = xr.where(self.cell.output.cftfrac.isel(time=-1) > 0, True, False).drop(  # noqa
            "time"
        )  # noqa
        self.crops = (
            self.cell.output.cftfrac.isel(time=[-1]).where(mask, drop=True).to_pandas()  # noqa
        )
        self.mem_fert = self.cell.output.cft_nfert.where(mask, drop=True).to_pandas()  # noqa
        self.mem_yield = self.cell.output.pft_harvestc.where(
            mask, drop=True
        ).to_pandas()  # noqa

        # init llm responses
        self.reasoning = None

    def get_response(self, messages):
        """Get response from llm"""
        success = False
        retry = 0
        max_retries = 30
        while retry < max_retries and not success:
            try:
                response = self.model.llm_client.chat.completions.create(
                    model=self.model.llm_name,
                    messages=messages,
                    # this is the degree of randomness of the model's output
                    temperature=self.model.temperature,
                )

                success = True
            except Exception as e:
                print(f"Error: {e} nRetrying ...")
                retry += 1
                time.sleep(0.5)

        if not success:
            raise Exception(
                f"Could not get response from llm after {max_retries} retries."
            )  # noqa

        return response.choices[0].message.content

    def decide_fertilizer(self):
        question_prompt = f"""
        You are a representative farmer of a whole simulation cell of 0.5x0.5°
        at {self.position} (cell centre).
        On your farm, you have the following crops (crop functional types) with
        the corresponding fraction of your land occupied by those crops:
        {self.crops}.
        Now, you are deciding on how much fertilizer you want to apply for each
        crop.
        Your task is to increase the crop yield by increasing the application
        of N fertilizer as long as it is reasonable. Also take into
        account the local conditions as well as available literature
        estimates for the optimal amount of fertilizer that should be applied
        to the crop.
        This also means that if higher fertilization does not increase the
        crop yield as before, you stop or even reduce the amount again.
        Here is the amount of fertilizer applied in the last years in
        Ng/m2/year: {self.mem_fert}.
        And here is the corresponding yield per crop from the last 10 years in
        gC/m2/year: {self.mem_yield}.
        Based on this context, you need to estimate the amount of fertilizer
        per crop in Ng/m2/year for the upcoming year.
        You must provide your reasoning (max 200 characters) for your choice
        and then your response by providing the amount of fertilizer you want
        to apply to each crop type.
        For example, if you would apply fertilizer amounts of 5 and 8
        (Ng/m2/year) to rainfed temperate cereals and irrigated biomass tree
        and no fertilizer to any other crop, your response will be:
        Reasoning: [Your reason to choose to apply less or more fertilizer]
        Response: [['rainfed temperate cereals,'irrigated biomass tree],[5,8]]
        Make sure your response is in this format.
        Also do not use semi-colons ";".
        """
        messages = [{"role": "system", "content": question_prompt}]
        try:
            output = self.get_response(messages)
        except Exception as e:
            raise Exception(f"Error: {e}\nCould not get response from llm.")

        try:
            intermediate = output.split("Reasoning:", 1)[1]
            reasoning, response = intermediate.split("Response:")
            self.reasoning = reasoning.strip()
            response = response.strip()
            response_data = ast.literal_eval(response)
            change_crops = response_data[0]
            change_fertilizer = response_data[1]

        except Exception as e:
            raise Exception(f"Error: {e}\nOutput: {output} could not be parsed.")  # noqa

        # set the new fertilizer values
        self.cell.input.fertilizer_nr.isel(time=-1).loc[{"band": change_crops}] = (  # noqa
            change_fertilizer  # noqa
        )

    def update(self, t):
        # call the base class update method
        super().update(t)
        self.decide_fertilizer()
