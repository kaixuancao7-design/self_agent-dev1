#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 QA 接口的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent.react_agent import ReactAgent
from utils.logger_handler import get_logger

logger = get_logger(__name__)

def test_qa():
    agent = ReactAgent()
    query = "AI Agent最近热点"
    
    logger.info(f"Testing query: {query}")
    
    response = ""
    try:
        for chunk in agent.execute_stream(query):
            response += chunk
            print(chunk, end="", flush=True)
        print("\n" + "="*50)
        print(f"Full response: {repr(response)}")
    except Exception as e:
        logger.error(f"Error during test: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_qa()