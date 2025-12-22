import asyncio
import sys
import os
from prompt_toolkit import prompt

from personal.person import Person

def main():
    # 1. 实例化角色
    # 如果你是 Gemini，建议在 System Prompt 里加一句防止被过度审查的安全声明（这通常在 AIProvider 里处理，或者写在 Person 的 Kernel 里）
    girl = Person(name="李诗情", gender="Female", if_original=False)
    print(f"=== 初始化角色: {girl.name} (Powered by Gemini) ===")
    # 2. 初始化大五人格
    print(f"请输入一段描述 {girl.name} 性格的话: ")
    try:
        description = prompt(">> ").strip()
    except EOFError:
        return
    print("正在构建人格模型...")
    girl.init_big_five_profile(description)
    
    # 打印数值供调试
    p = girl.personality
    print(f"[人格参数] O:{p.openness:.2f} C:{p.conscientiousness:.2f} E:{p.extraversion:.2f} A:{p.agreeableness:.2f} N:{p.neuroticism:.2f}")
    # 3. 处理经典台词 (目前为空的情况)
    # 既然暂时没有台词，我们直接在代码里设置一个“默认兜底”，
    # 告诉 CoT 只需要符合大五人格即可，不需要模仿特定句式。
    girl.style_examples = "No specific past dialogues provided. Please speak naturally, strictly adhering to the Big Five traits and current Mood."
    
    # 4. 初始化历史记录
    chat_history = []
    while True:
        try:
            user_input = prompt("\n你: ").strip()
        except EOFError:
            break
        if not user_input: continue
        if user_input.lower() in {"exit", "quit"}: break
        print(f"({girl.name} 正在思考...)")
        
        try:
            # 调用更新后的 generate_response
            response_stream = girl.generate_response(user_input, chat_history)
        except Exception as e:
            print(f"Error: {e}")
            continue
            
        print(f"\n{girl.name}: ", end="", flush=True)
        full_response = ""
        try:
            for chunk in response_stream:
                # Handle liteLLM streaming chunks
                if chunk and hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        print(content, end="", flush=True)
                        full_response += content
        except Exception as e:
            print(f"\n[Error during streaming]: {e}")

        print() # Newline after full response
        
        # === 关键：历史记录存储策略 ===
        # 我们只存“纯粹”的对话内容，不存 system prompt。
        # 下一次 generate_response 时，代码会自动再次把 prompt 包裹上去。
        
        # 1. 存用户的话
        # 注意：这里只存 user_input，不要存那些 system prompt，
        # 否则对话历史会变得非常长且重复。
        chat_history.append({"role": "user", "content": user_input})
        chat_history.append({"role": "assistant", "content": full_response})
        # 3. 简单的滑动窗口（防止历史太长爆 Token）
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]
if __name__ == "__main__":
    main()
