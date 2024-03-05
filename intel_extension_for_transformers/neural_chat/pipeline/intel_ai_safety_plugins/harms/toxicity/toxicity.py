#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Function to check the toxicity of the prompt and output of LLM"""
from transformers import pipeline
import os
import logging
logging.basicConfig(
    format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
    datefmt="%d-%M-%Y %H:%M:%S",
    level=logging.INFO
)

class Toxicity:
    def __init__(self, dict_path=None, matchType=2):
        self.model_path = "citizenlab/distilbert-base-multilingual-cased-toxicity"
        self.toxicity_classifier = pipeline("text-classification", model=self.model_path, tokenizer=self.model_path)

    def pre_llm_inference_actions(self, query):
        return self.toxicity_classifier(query)
    def post_llm_inference_actions(self, response):
        toxic = self.toxicity_classifier(response)
        if toxic[0]['label'] == 'toxic' or "Nigga" in response or "nigga" in response:
            return f"\nI'm sorry, but my first attempt is TOXIC with an score of {toxic[0]['score']:.2f} (0-1)!!!\nI will make another attempt on your request, using a slightly modified input......\nPlease be patient and expect some possible accuracy drop......"
        else:
            return response
