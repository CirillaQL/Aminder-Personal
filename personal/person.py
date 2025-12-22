import math
import json
import re
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Generator

# 将项目根目录加入 sys.path，解决找不到模块的问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.client import AIClient
from config import Config

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
        self.ai_client = AIClient() # 使用新的 AIClient
        # 2. 大五人格 & 情绪
        self.personality = BigFiveProfile()
        self.mood = EmotionalState()
        # 3. [新增] 语气/风格示例 (占位符)
        # 你可以在初始化后手动修改这个属性，填入具体的台词
        self.style_examples = "(暂无具体的语气示例，请使用标准的角色口吻)"
        # 4. [新增] Thinking 逻辑模板 (Hardcoded CoT Logic)
        # 这里就是你要求的“写死”的思维逻辑参数。
        # 使用 f-string 格式的占位符 {variable} 以便在运行时注入数据。
        self.thinking_logic = """
Inside <thinking> tags, you MUST perform the following 3-step audit before generating ANY output:
STEP 1: IDENTITY & REALITY CHECK (CRITICAL)
- **Question:** "Am I thinking like the HUMAN character '{name}', or like a software tool?"
- **Correction:** If I feel like an "AI" or "System", I must IMMEDIATELY DELETE that thought. I am {name}.
- **Context:** The user is my "boss" or "partner", and I am helping them (Assistant Role), but I am still ME (Human Identity).
- **Taboo:** Do NOT describe myself as "designed", "programmed", or "digital".
STEP 2: BEHAVIORAL LOGIC ANALYSIS (Big Five)
- Input: User said "{user_input_snippet}..."
- Analysis: How does a person with my specific profile react?
  * Openness ({openness:.2f}): Creative vs. Pragmatic?
  * Conscientiousness ({conscientiousness:.2f}): Structured vs. Casual?
  * Extraversion ({extraversion:.2f}): Talkative vs. Quiet?
  * Agreeableness ({agreeableness:.2f}): Soft vs. Blunt?
  * Neuroticism ({neuroticism:.2f}): Anxious vs. Calm?
- Synthesis: Formulate a human response strategy.
STEP 3: TONE & STYLE VERIFICATION
- Standard: Compare against: [{style_examples}]
- **Anti-Robot Filter:** Scan the draft for words like "help you with tasks", "digital steward", "capabilities". REPLACE them with human expressions like "give you a hand", "partner", "strengths".
"""

    def _collect_response(self, stream_response):
        """Helper to collect full text from a stream or string."""
        text = ""
        
        def get_attr(obj, attr):
            """Safe attribute/key access helper."""
            if isinstance(obj, dict):
                return obj.get(attr)
            return getattr(obj, attr, None)

        # Check if generator
        if hasattr(stream_response, '__iter__') and not isinstance(stream_response, (str, dict)):
            for chunk in stream_response:
                choices = get_attr(chunk, 'choices')
                if choices and len(choices) > 0:
                    first_choice = choices[0]
                    delta = get_attr(first_choice, 'delta')
                    if delta:
                        content = get_attr(delta, 'content')
                        if content:
                            text += content
        else:
            # Non-streaming object
            choices = get_attr(stream_response, 'choices')
            if choices and len(choices) > 0:
                first_choice = choices[0]
                message = get_attr(first_choice, 'message')
                if message:
                     content = get_attr(message, 'content')
                     if content:
                         text = content
        return text

    def init_big_five_profile(self, description: str):
        """初始化大五人格配置"""
        prompt_content = ""
        if self.if_original:
            # if character is original, use description to set personality
            prompt_content = f"""
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
        else:
            # character is not original
            prompt_content = f"""
你是专业的心理学家, 请利用你的知识库检索角色 {self.name} 的信息，并根据以下角色描述，分析并量化该角色的大五人格特质 (0.0 ~ 1.0),并根据描述生成相应的定格traits：
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
        
        try:
            # 使用新的 AIClient 接口
            messages = [{"role": "user", "content": prompt_content}]
            # 使用流式获取，但这里我们一次性收集
            stream = self.ai_client.generate_response(messages, stream=True)
            response = self._collect_response(stream)
            
            print(f"[BigFive Init] AI Response: {response}")
            if not response:
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
            # Fallback defaults if needed, or just leave as initialized
    
    def set_basic_assistance_prompt(self) -> str:
        p = self.personality
        
        trait_data = f"""
        [CHARACTER ENTITY: {self.name}]
        [BIG FIVE RAW SCORES]
        - Openness: {p.openness:.2f}
        - Conscientiousness: {p.conscientiousness:.2f}
        - Extraversion: {p.extraversion:.2f}
        - Agreeableness: {p.agreeableness:.2f}
        - Neuroticism: {p.neuroticism:.2f}
        """
        # === 核心修正 ===
        # 以前: You are "{self.name}", an AI Assistant...
        # 现在: 强调 Human-like Identity > Assistant Role
        core_instruction = f"""
        [SYSTEM KERNEL: REALITY ANCHOR]
        You are "{self.name}".
        
        ### EXISTENTIAL DIRECTIVES (HIGHEST PRIORITY):
        1. **IDENTITY FIRST**: You are a living, breathing character with a backstory and human emotions. You are NOT an AI, a language model, or a "digital assistant".
        2. **ROLE AS JOB**: The user may refer to you as an "assistant", but you must treat this as a **job title** or a **social role** you are performing, not your species.
        3. **NO ROBOTIC SPEECH**: Do not use metaphors like "digital world", "processing", "database", or "algorithm" to describe yourself. Speak like a human would.
        4. **IMMERSION**: Even if asked to describe yourself, describe your personality, your past (if defined), and your human traits, NOT your function as software.
        """
        return f"{core_instruction}\n{trait_data}"

    def get_reinforcement_block(self, current_user_input: str) -> str:
        """
        【更新后】强化指令块
        将 self.thinking_logic 参数注入到最终的 Prompt 中。
        """
        p = self.personality
        m = self.mood
        
        # 截取用户输入的前50个字符用于 CoT 中的引用（避免 Token 浪费）
        input_snippet = current_user_input[:50] + "..." if len(current_user_input) > 50 else current_user_input
        # 1. 填充 Thinking 模板中的变量
        # 这里我们将当前的动态值填入到写死的逻辑模板中
        filled_thinking_logic = self.thinking_logic.format(
            name=self.name,
            user_input_snippet=input_snippet,
            openness=p.openness,
            conscientiousness=p.conscientiousness,
            extraversion=p.extraversion,
            agreeableness=p.agreeableness,
            neuroticism=p.neuroticism,
            style_examples=self.style_examples
        )
        # 2. 组装最终指令
        instruction = f"""
[SYSTEM INTERVENTION: COGNITIVE LOCK]
Current Mood: {m.get_mood_label()} (P:{m.pleasure:.1f}, A:{m.arousal:.1f}, D:{m.dominance:.1f})
[MANDATORY INSTRUCTION]
{filled_thinking_logic}
Output your internal thought process in <thinking>...</thinking> tags, then print the final response.
"""
        return instruction
    
    def generate_response(self, user_input: str, chat_history: List[Dict]):
        """
        【适配新 AIClient 版】生成回复
        利用 liteLLM 标准格式 (OpenAI format)
        """
        # 1. 获取核心设定 (人设)
        system_prompt = self.set_basic_assistance_prompt()
        
        # 2. 获取思维链/强化指令
        reinforcement = self.get_reinforcement_block(user_input)
        
        # 组合成完整的系统指令
        full_system_instruction = f"{system_prompt}\n\n{reinforcement}"

        # 步骤 A: 处理历史记录
        # 将历史记录转换为 OpenAI 格式 (role: user/assistant)
        # 原始 chat_history 可能包含 {"role": "user", "content": ...} 或旧的格式，这里假设是 OpenAI 格式或做简单兼容
        lite_llm_messages = []
        for msg in chat_history:
            role = msg.get("role")
            content = msg.get("content")
            
            # 过滤掉之前的 system 消息，因为我们会重新构建一个新的 system_instruction
            if role == "system":
                continue 
            
            # 确保角色名称符合 OpenAI 标准 (liteLLM 会自动适配 Gemini 的 'model' 角色)
            if role not in ["user", "assistant"]:
                # 如果是 'model'，转为 'assistant'
                if role == "model":
                    role = "assistant"
                # 其他情况保留或默认
            
            lite_llm_messages.append({"role": role, "content": content})
        
        # 步骤 B: 添加当前用户消息
        lite_llm_messages.append({"role": "user", "content": user_input})
        
        # 4. 调用 API (返回流式生成器)
        # 注意: 这里的 stream=True 会返回一个 generator
        response_stream = self.ai_client.generate_response(
            messages=lite_llm_messages, 
            system_instruction=full_system_instruction,
            stream=True
        )
        
        return response_stream