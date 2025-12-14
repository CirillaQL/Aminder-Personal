import asyncio
import os
import sys
from personal.personal import Person, BigFiveProfile


# Ensure we can import from personal if not in path
sys.path.append(os.getcwd())

async def main():
    print("Initializing AI Assistant...")
    
    # Create a BigFiveProfile for a gentle and kind personality
    gentle_personality = BigFiveProfile(
        openness=0.7,          # Imaginative, open to new ideas
        conscientiousness=0.6, # Organized, diligent
        extraversion=0.6,      # Friendly, outgoing
        agreeableness=0.8,     # Kind, compassionate, cooperative
        neuroticism=0.2        # Emotionally stable, calm
    )

    # Create a Person instance
    assistant_name = "赵今麦" # Example name
    assistant_gender = "Female" # Example gender
    try:
        assistant = Person(
            name=assistant_name, 
            gender=assistant_gender, 
            personality=gentle_personality
        )
        print(f"AI Assistant '{assistant_name}' initialized with a gentle and kind personality.")
        print(f"--- Initial State for {assistant_name} ---")
        print(assistant.get_system_prompt_context())
        print("Chat initialized. Type 'exit' or 'quit' to stop.")
    except Exception as e:
        print(f"Failed to initialize AI Assistant: {e}")
        return

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            # Simulate simple emotional stimulus based on keywords (for testing personality interaction)
            if "bad" in user_input.lower() or "hate" in user_input.lower():
                print("[System: Detected negative stimulus for assistant]")
                assistant.receive_stimulus(-0.3, 0.5) # Slight negative impact
            elif "good" in user_input.lower() or "love" in user_input.lower():
                print("[System: Detected positive stimulus for assistant]")
                assistant.receive_stimulus(0.3, 0.2) # Slight positive impact

            response = await assistant.chat(user_input)
            print(f"{assistant.name}: {response}")
            
            # AI Driven State Update (commented out for now to simplify initial testing, uncomment later)
            await assistant.update_state_via_ai(user_input, response)
            print(f"[State Updated] Mood: {assistant.mood.get_mood_label()} | Traits: {assistant.personality.get_description()}")
            
            # Simulate time passing for emotional decay
            assistant.tick(0.1)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
