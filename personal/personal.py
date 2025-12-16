import math
import json
import re
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# 将项目根目录加入 sys.path，解决找不到 chat 模块的问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chat.client import AIProvider

@dataclass
class BigFiveProfile:
    """
    大五人格模型 (0.0 ~ 1.0)
    """
    openness: float = 0.5          # 开放性 (O)
    conscientiousness: float = 0.5 # 尽责性 (C)
    extraversion: float = 0.5      # 外向性 (E)
    agreeableness: float = 0.5     # 宜人性 (A)
    neuroticism: float = 0.5       # 神经质 (N)
    
    # AI 生成的特征标签
    traits: List[str] = field(default_factory=list)

# --- 2. 动态状态层: PAD 三维情绪模型 ---
@dataclass
class EmotionalState:
    """
    PAD 情绪模型 (-1.0 ~ 1.0)
    Pleasure (愉悦度): 不爽 <-> 爽
    Arousal  (激活度): 困倦/平静 <-> 激动/警惕
    Dominance(优势度): 顺从/恐惧 <-> 掌控/自信
    """
    pleasure: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    
    # 能量值 (0.0 ~ 1.0)，模拟疲劳
    energy: float = 1.0 

    def update(self, d_p: float, d_a: float, d_d: float):
        """情绪受到刺激后的变化"""
        self.pleasure = max(-1.0, min(1.0, self.pleasure + d_p))
        self.arousal = max(-1.0, min(1.0, self.arousal + d_a))
        self.dominance = max(-1.0, min(1.0, self.dominance + d_d))

    def decay(self, rate: float = 0.1):
        """
        情绪衰减机制 (回归平静)
        随着时间流逝，情绪会趋向于 0 (平静)
        """
        self.pleasure *= (1 - rate)
        self.arousal *= (1 - rate)
        # Dominance 通常比较稳定，衰减稍慢
        self.dominance *= (1 - rate * 0.5)

    def get_mood_label(self) -> str:
        """
        将 PAD 数值映射为离散的情绪标签 (用于注入 Prompt)
        这是一个简化版的映射逻辑
        """
        P, A, D = self.pleasure, self.arousal, self.dominance
        
        if A < 0 and P > 0: return "Relaxed (惬意放松)"
        if A < 0 and P < 0: return "Bored/Depressed (无聊/沮丧)"
        
        if A > 0:
            if P > 0.5 and D > 0: return "Joyful (兴高采烈)"
            if P > 0.2 and D > 0: return "Excited (兴奋)"
            if P < -0.5 and D > 0: return "Angry (愤怒)"  # 不爽+强势
            if P < -0.5 and D < 0: return "Fearful (恐惧)" # 不爽+弱势
            if P < 0: return "Anxious (焦虑)"
            
        return "Neutral (平静)"

# --- 3. 核心实体类 ---
class Person(object):
    def __init__( 
                 self, 
                 name: str, 
                 gender: str, 
                 if_original: bool = False,
                 ):
        # 1. 基础信息
        self.name = name
        self.if_original = if_original
        self.gender = gender
        self.ai_client = AIProvider()

        # 2. 大五人格
        self.personality = BigFiveProfile()
        
        # 3. 动态状态 (State) - 初始为平静
        self.mood = EmotionalState()
        
        # 4. 记忆与知识
        # self.memory_engine = MemoryEngine() # 暂时注释掉，因为 MemoryEngine 未定义

    def init_big_five_profile(self, description: str):
        """初始化大五人格配置"""
        if self.if_original:
            # if character is original, use description to set personality
            description_prompt = f"""
你是专业的心理学家，请根据以下角色描述，分析并量化该角色的大五人格特质 (0.0 ~ 1.0),并根据描述生成相应的定格traits：
角色描述: {description}
请只返回一个 JSON 格式的文本，格式如下，不要添加任何其他内容：
{{
  "openness": float,
  "conscientiousness": float,
  "extraversion": float,
  "agreeableness": float,
  "neuroticism": float,
  "traits": ["trait1", "trait2", ...]
}}
"""
            response = self.ai_client.send_message(description_prompt, web_search=False)
            
            try:
                print(f"[BigFive Init] AI Response: {response}")
                if response is None:
                    raise ValueError("No response from AI client.")
                # 移除可能存在的 markdown 代码块标记 (```json ... ```)
                cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned_response)
                self.personality.openness = max(0.0, min(1.0, data.get("openness", 0.5)))
                self.personality.conscientiousness = max(0.0, min(1.0, data.get("conscientiousness", 0.5)))
                self.personality.extraversion = max(0.0, min(1.0, data.get("extraversion", 0.5)))
                self.personality.agreeableness = max(0.0, min(1.0, data.get("agreeableness", 0.5)))
                self.personality.neuroticism = max(0.0, min(1.0, data.get("neuroticism", 0.5)))
                self.personality.traits = data.get("traits", [])
            except Exception as e:
                print(f"[BigFive Init Error] {e}")
        else:
            # character is not original, so we need to search web and use description to set personality.
            description_prompt = f"""
你是专业的心理学家, 请联网检索角色 {self.name} 的信息，并根据以下角色描述，分析并量化该角色的大五人格特质 (0.0 ~ 1.0),并根据描述生成相应的定格traits：
角色描述: {description}
请只返回一个 JSON 格式的文本，格式如下，不要添加任何其他内容：
{{
  "openness": float,
  "conscientiousness": float,
  "extraversion": float,
  "agreeableness": float,
  "neuroticism": float,
  "traits": ["trait1", "trait2", ...]
}}
"""
            response = self.ai_client.send_message(description_prompt, web_search=True)
            try:
                print(f"[BigFive Init] AI Response: {response}")
                if response is None:
                    raise ValueError("No response from AI client.")
                # 移除可能存在的 markdown 代码块标记 (```json ... ```)
                cleaned_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned_response)
                self.personality.openness = max(0.0, min(1.0, data.get("openness", 0.5)))
                self.personality.conscientiousness = max(0.0, min(1.0, data.get("conscientiousness", 0.5)))
                self.personality.extraversion = max(0.0, min(1.0, data.get("extraversion", 0.5)))
                self.personality.agreeableness = max(0.0, min(1.0, data.get("agreeableness", 0.5)))
                self.personality.neuroticism = max(0.0, min(1.0, data.get("neuroticism", 0.5)))
                self.personality.traits = data.get("traits", [])
            except Exception as e:
                print(f"[BigFive Init Error] {e}")