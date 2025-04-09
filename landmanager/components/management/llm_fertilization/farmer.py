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
        You are a representative farmer for the 0.5°x0.5° simulation cell
        centered at {self.position}.

        You manage the following crop types on your land (with land fractions):
        {self.crops}

        Your goal: **maximize crop yield** by adjusting nitrogen (N) fertilizer
        application per crop (in gN/m²/year).

        Consider the following historical data as your memory (last 9 years):
        - Fertilizer application: {self.mem_fert}
        - Crop yields (in gC/m²/year): {self.mem_yield}

        Apply the following rules:
        1. Any crop currently fertilized with **less than 5 gN/m²/year** is
        likely underfertilized. **Increase to crop-specific levels seen in
        intensive agriculture (~10–30 gN/m²/year)**
        2. You have been responsible for fertilizer decisions over the last
        10 years. Your memory covers the past 9 years of data. This year
        completes the full 10-year span. All rules defined here apply over this
        period.
        3. Yield responses may take time. **Maintain or steadily increase**
        fertilizer for 3–5 years to observe long-term effects.
        4. If yields have not clearly saturated (based on memory), continue
        **increasing fertilizer by 3–5 gN/m²/year** to explore the response.
        5. **Never apply more than 3 gN/m²/year to nitrogen-fixing crops**
        (e.g., pulses, soybean). This is a hard upper limit. Excess N reduces
        their yield.
        6. Only **reduce fertilizer** if there is a clear and consistent
        **negative yield trend over the past 3–5 years**. This is only relevant
        for nitrogen-fixing crops.

        Your task:
        Estimate the fertilizer amount (gN/m²/year) for each crop for the
        coming year, following all rules equally.

        ### Format your response exactly like this:
        Reasoning: [Max 200 characters]
        Response: [['crop1','crop2'], [amount1, amount2]]

        Example:
        Reasoning: Increased temperate cereal N due to low yield response,
        kept others constant.
        Response: [['rainfed temperate cereals','irrigated biomass tree'], [5, 8]]

        Technical Note:
        - Use **only** the format above.
        - Ensure that crop names and amounts match and that both lists are the
        same length.
        - Do **not** use semicolons ";" in your response.
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
