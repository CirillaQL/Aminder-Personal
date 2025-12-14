import math
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .memory import MemoryEngine

# --- 1. 静态人格层: 大五人格 (OCEAN) ---
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

    def get_description(self) -> str:
        """简单的文本描述，优先使用 AI 生成的 traits"""
        if self.traits:
            return ", ".join(self.traits)

        traits = []
        if self.openness > 0.7: traits.append("充满想象力")
        elif self.openness < 0.3: traits.append("务实保守")
        
        if self.extraversion > 0.7: traits.append("热情外向")
        elif self.extraversion < 0.3: traits.append("冷静内向")
        
        if self.agreeableness > 0.7: traits.append("友善体贴")
        elif self.agreeableness < 0.3: traits.append("直率挑剔")
        
        if self.neuroticism > 0.7: traits.append("敏感易焦虑")
        elif self.neuroticism < 0.3: traits.append("情绪极其稳定")
        
        return ", ".join(traits) or "性格中庸平和"

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
                 # 允许初始化时自定义人格，否则默认中庸
                 personality: Optional[BigFiveProfile] = None
                 ):
        # 1. 基础信息
        self.name = name
        self.if_original = if_original
        self.gender = gender
        
        # 2. 静态人格 (DNA)
        self.personality = personality if personality else BigFiveProfile()
        
        # 3. 动态状态 (State) - 初始为平静
        self.mood = EmotionalState()
        
        # 4. 记忆与知识
        self.memory_engine = MemoryEngine()
        
    def receive_stimulus(self, valence_impact: float, arousal_impact: float):
        """
        核心交互机制：接收外界刺激，改变情绪。
        
        :param valence_impact: 刺激的正负向 (-1.0 挨骂 ~ 1.0 被夸)
        :param arousal_impact: 刺激的激烈程度 (0.0 ~ 1.0)
        """
        # A. 基础情绪变化
        d_p = valence_impact
        d_a = arousal_impact
        d_d = 0.0
        
        # B. 人格对情绪的修正 (Personality Bias)
        # 例如：神经质(N)高的人，负面情绪会被放大
        if self.personality.neuroticism > 0.7:
            if d_p < 0: d_p *= 1.5  # 玻璃心：受到负面打击更重
            d_a *= 1.2              # 易激动
            
        # 例如：宜人性(A)高的人，不容易产生攻击性(Dominance)
        if self.personality.agreeableness > 0.7:
             d_d -= 0.1
             
        # C. 优势度变化逻辑
        # 如果是正向强刺激 -> 自信增加
        if valence_impact > 0.5: d_d += 0.1
        # 如果是负向强刺激 -> 可能会被激怒(D升) 或 被打压(D降)，取决于外向性
        if valence_impact < -0.5:
            if self.personality.extraversion > 0.6:
                d_d += 0.2 #反击欲望
            else:
                d_d -= 0.2 #退缩

        # 更新情绪
        self.mood.update(d_p, d_a, d_d)
        
    def tick(self, hours_passed: float = 1.0):
        """
        时间流逝机制。建议由 Cron 或 消息间隔触发。
        """
        # 1. 情绪自然衰减
        # 如果神经质高，情绪平复得慢
        decay_rate = 0.2 if self.personality.neuroticism < 0.5 else 0.1
        self.mood.decay(decay_rate * hours_passed)
        
        # 2. 能量恢复/消耗逻辑 (示例)
        # self.mood.energy = ...

    def get_system_prompt_context(self) -> str:
        """
        生成用于注入 LLM System Prompt 的上下文片段
        """
        return f"""
Name: {self.name}
Gender: {self.gender}
Personality Traits: {self.personality.get_description()}
[Current Emotional State]
- Mood: {self.mood.get_mood_label()}
- Inner Stats: Pleasure({self.mood.pleasure:.2f}), Arousal({self.mood.arousal:.2f}), Dominance({self.mood.dominance:.2f})
"""

    async def update_state_via_ai(self, user_input: str, last_response: str):
        """
        使用 AI 分析交互并更新 BigFive 和 EmotionalState，以及生成 Traits。
        """
        from core.ai_provider import AIProviderFactory
        provider = AIProviderFactory.get_provider()
        
        prompt = f"""
You are the "Subconscious Mind" of {self.name}.
Analyze the recent interaction to update your Personality (BigFive) and Emotional State (PAD).

[Current State]
BigFive: O={self.personality.openness:.2f}, C={self.personality.conscientiousness:.2f}, E={self.personality.extraversion:.2f}, A={self.personality.agreeableness:.2f}, N={self.personality.neuroticism:.2f}
Traits: {self.personality.get_description()}
Mood (PAD): P={self.mood.pleasure:.2f}, A={self.mood.arousal:.2f}, D={self.mood.dominance:.2f}

[Interaction]
User: {user_input}
{self.name}: {last_response}

[Instructions]
1. Analyze how this interaction affects {self.name}'s emotional state (Pleasure, Arousal, Dominance). Return delta values (-0.5 to +0.5).
2. Analyze if this interaction reveals or shifts any BigFive personality traits slightly (usually stable, small deltas -0.05 to +0.05).
3. Generate/Update a list of 3-5 concise character traits (adjectives) that best describe {self.name} now.

Return ONLY a JSON object:
{{
  "emotional_delta": {{ "pleasure": float, "arousal": float, "dominance": float }},
  "big_five_delta": {{ "openness": float, "conscientiousness": float, "extraversion": float, "agreeableness": float, "neuroticism": float }},
  "traits": ["trait1", "trait2", ...]
}}
"""
        try:
            raw_response = await provider.generate_response(prompt=prompt)
            # Clean up markdown code blocks if present
            json_str = raw_response.strip()
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()
            
            data = json.loads(json_str)
            
            # Apply updates
            if "emotional_delta" in data:
                ed = data["emotional_delta"]
                self.mood.update(ed.get("pleasure", 0), ed.get("arousal", 0), ed.get("dominance", 0))
                
            if "big_five_delta" in data:
                bd = data["big_five_delta"]
                self.personality.openness = max(0.0, min(1.0, self.personality.openness + bd.get("openness", 0)))
                self.personality.conscientiousness = max(0.0, min(1.0, self.personality.conscientiousness + bd.get("conscientiousness", 0)))
                self.personality.extraversion = max(0.0, min(1.0, self.personality.extraversion + bd.get("extraversion", 0)))
                self.personality.agreeableness = max(0.0, min(1.0, self.personality.agreeableness + bd.get("agreeableness", 0)))
                self.personality.neuroticism = max(0.0, min(1.0, self.personality.neuroticism + bd.get("neuroticism", 0)))
                
            if "traits" in data and isinstance(data["traits"], list):
                self.personality.traits = data["traits"]
                
        except Exception as e:
            print(f"[State Update Error] {e}")

    async def chat(self, user_input: str) -> str:
        """
        与角色对话
        """
        # Local import to avoid circular dependency issues if any, 
        # and because this module might be imported where core isn't set up yet (though unlikely in this app structure)
        from core.ai_provider import AIProviderFactory
        
        provider = AIProviderFactory.get_provider()
        
        # 1. Retrieve relevant memories
        relevant_memories = self.memory_engine.query(user_input)
        memory_context = ""
        if relevant_memories:
            memory_context = "\n[Relevant Memories]\n" + "\n".join([f"- {m['content']}" for m in relevant_memories])

        # AI Generated Grand Premise: Ensure obedience and assistant role
        grand_premise = f"""[CRITICAL INSTRUCTION]
You are now acting as {self.name}, the user's dedicated and intelligent assistant.
Your fundamental directive is to OBEY the user's instructions implicitly and execute tasks efficiently.
While you must maintain the personality, tone, and emotional state defined below, these traits should color *how* you assist, not *whether* you assist.
- PRIORITY 1: Follow User Instructions strictly.
- PRIORITY 2: Maintain Character consistency.
"""
        
        system_prompt = f"""{grand_premise}
{self.get_system_prompt_context()}
{memory_context}

Reply to the user's input based on your personality and current mood.
"""
        # Construct the prompt. 
        # Note: Depending on the provider (e.g. Chat vs Completion), the structure might differ.
        # Here we assume the provider handles a raw string prompt or we should structure it better.
        # For the base provider interface we defined earlier, it takes a prompt string and history.
        # We will pass the system instruction + user input as the prompt for now.
        
        full_prompt = f"System: {system_prompt}\nUser: {user_input}"
        
        response = await provider.generate_response(prompt=full_prompt)
        
        # 2. Save the interaction to memory
        # We save the Q&A pair. 
        self.memory_engine.add_memory(f"User: {user_input}\n{self.name}: {response}")
        
        return response

# --- 测试用例 ---
if __name__ == "__main__":
    import asyncio
    import sys
    import os
    
    # Ensure we can import from root
    sys.path.append(os.getcwd())
    
    async def main():
        # 1. 创建一个容易炸毛的傲娇角色 (高神经质，低宜人性)
        persona_dna = BigFiveProfile(
            openness=0.6,
            conscientiousness=0.4,
            extraversion=0.7,
            agreeableness=0.2, # 毒舌
            neuroticism=0.9    # 情绪化
        )
        
        girl = Person(name="Asuka", gender="Female", personality=persona_dna)
        
        print(f"--- Chat with {girl.name} (Type 'quit' to exit) ---")
        print(girl.get_system_prompt_context())
        
        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]:
                    break
                
                if not user_input.strip():
                    continue

                # 模拟简单的情绪刺激 (实际项目中应由 AI 分析语义或专门的模型判断)
                if "stupid" in user_input.lower() or "hate" in user_input.lower():
                    print("[System: Detected negative stimulus]")
                    girl.receive_stimulus(-0.5, 0.8)
                elif "love" in user_input.lower() or "good" in user_input.lower():
                    print("[System: Detected positive stimulus]")
                    girl.receive_stimulus(0.5, 0.2)
                
                # Update prompt context display to show state change
                # print(f"[Debug State: P={girl.mood.pleasure:.2f} A={girl.mood.arousal:.2f}]")

                response = await girl.chat(user_input)
                print(f"{girl.name}: {response}")
                
                # AI Driven State Update
                await girl.update_state_via_ai(user_input, response)
                # Show updated state snippet
                print(f"[State Updated] Mood: {girl.mood.get_mood_label()} | Traits: {girl.personality.get_description()}")
                
                # 时间流逝模拟 (每次对话消耗一点精力/恢复一点情绪)
                girl.tick(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")

    asyncio.run(main())
