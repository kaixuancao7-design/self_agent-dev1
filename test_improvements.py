#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试所有改进功能
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, '.')

def test_rag_module():
    """测试RAG模块"""
    print('=== 测试RAG模块 ===')
    try:
        from rag.vector_store import VectorStoreService, ChromaCacheManager
        from rag.hybrid_retriever import HybridRetriever
        
        print('  - 测试ChromaCacheManager...')
        cache_manager = ChromaCacheManager('/tmp/test_cache')
        cache_manager.set_cache('test_db', 'test_query', ['result1', 'result2'])
        cached = cache_manager.get_cache('test_db', 'test_query')
        assert cached == ['result1', 'result2'], '缓存设置/获取失败'
        print('    OK 缓存管理器测试通过')
        
        print('  - 测试空Chunk过滤...')
        from langchain_core.documents import Document
        docs = [
            Document(page_content=''),
            Document(page_content='   '),
            Document(page_content='short'),
            Document(page_content='this is a valid document with enough content')
        ]
        vs = VectorStoreService.__new__(VectorStoreService)
        filtered = vs._filter_empty_chunks(docs)
        assert len(filtered) == 1, '空Chunk过滤失败'
        print('    OK 空Chunk过滤测试通过')
        
        print('  OK RAG模块测试通过')
        return True
    except Exception as e:
        print('  FAIL RAG模块测试失败:', str(e)[:100])
        import traceback
        traceback.print_exc()
        return False


def test_hybrid_retriever():
    """测试混合检索器"""
    print('=== 测试混合检索器 ===')
    try:
        from rag.hybrid_retriever import HybridRetriever, get_hybrid_retriever
        
        print('  - 测试权重配置...')
        from utils.config_handler import rag_cfg
        retrieval_cfg = rag_cfg.get('retrieval', {})
        assert 'bm25_weight' in retrieval_cfg, 'bm25_weight配置缺失'
        assert 'vector_weight' in retrieval_cfg, 'vector_weight配置缺失'
        assert 'hybrid_weight' in retrieval_cfg, 'hybrid_weight配置缺失'
        assert 'rrf_k' in retrieval_cfg, 'rrf_k配置缺失'
        print('    OK 权重配置测试通过')
        
        print('  OK 混合检索器测试通过')
        return True
    except Exception as e:
        print('  FAIL 混合检索器测试失败:', str(e)[:100])
        import traceback
        traceback.print_exc()
        return False


def test_langgraph_workflow():
    """测试LangGraph工作流"""
    print('=== 测试LangGraph工作流 ===')
    try:
        from agent.langgraph_workflow import LangGraphAgent, AgentState
        
        print('  - 测试AgentState定义...')
        state = AgentState(
            messages=[],
            step_count=0,
            retry_count=0,
            is_finished=False,
            last_error=None
        )
        assert 'step_count' in state, 'step_count缺失'
        assert 'retry_count' in state, 'retry_count缺失'
        assert 'last_error' in state, 'last_error缺失'
        assert state['step_count'] == 0, 'step_count默认值错误'
        assert state['retry_count'] == 0, 'retry_count默认值错误'
        assert state['last_error'] is None, 'last_error默认值错误'
        print('    OK AgentState测试通过')
        
        print('  - 测试最大步数限制...')
        agent = LangGraphAgent(max_steps=5)
        assert agent.max_steps == 5, 'max_steps配置失败'
        assert agent.max_retries == 3, 'max_retries默认值错误'
        print('    OK 最大步数限制测试通过')
        
        print('  OK LangGraph工作流测试通过')
        return True
    except Exception as e:
        print('  FAIL LangGraph工作流测试失败:', str(e)[:100])
        import traceback
        traceback.print_exc()
        return False


def test_skill_loader():
    """测试SkillLoader"""
    print('=== 测试SkillLoader ===')
    try:
        from agent.skills.skill_loader import SkillLoader, FallbackSkill
        
        print('  - 测试FallbackSkill...')
        fallback = FallbackSkill('test_skill', Exception('测试错误'))
        result = fallback.invoke({'query': 'test'})
        assert 'test_skill' in result, '降级技能未返回正确消息'
        assert '测试错误' in result, '错误信息未包含在回复中'
        print('    OK FallbackSkill测试通过')
        
        print('  - 测试SkillLoader初始化...')
        loader = SkillLoader(enable_fallback=True)
        assert loader.enable_fallback == True, 'enable_fallback配置失败'
        skills = loader.load_all_skills()
        print('    发现', len(skills), '个技能')
        print('    OK SkillLoader测试通过')
        
        print('  OK SkillLoader测试通过')
        return True
    except Exception as e:
        print('  FAIL SkillLoader测试失败:', str(e)[:100])
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置文件"""
    print('=== 测试配置文件 ===')
    try:
        from utils.config_handler import rag_cfg, chroma_cfg
        
        print('  - 测试RAG配置...')
        retrieval = rag_cfg.get('retrieval', {})
        assert 'hybrid_weight' in retrieval, 'hybrid_weight缺失'
        assert 'bm25_k1' in retrieval, 'bm25_k1缺失'
        assert 'bm25_b' in retrieval, 'bm25_b缺失'
        assert 'fusion_method' in retrieval, 'fusion_method缺失'
        print('    OK RAG配置测试通过')
        
        print('  - 测试Chroma配置...')
        assert 'chunk_size' in chroma_cfg, 'chunk_size缺失'
        assert 'chunk_overlap' in chroma_cfg, 'chunk_overlap缺失'
        print('    OK Chroma配置测试通过')
        
        print('  OK 配置文件测试通过')
        return True
    except Exception as e:
        print('  FAIL 配置文件测试失败:', str(e)[:100])
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print('=' * 60)
    print('测试所有改进功能')
    print('=' * 60)
    print()
    
    tests = [
        test_config,
        test_rag_module,
        test_hybrid_retriever,
        test_langgraph_workflow,
        test_skill_loader
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        if test():
            passed += 1
        else:
            failed += 1
        print()
    
    print('=' * 60)
    print('测试结果:', passed, '通过,', failed, '失败')
    print('=' * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
