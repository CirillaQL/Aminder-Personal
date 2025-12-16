import asyncio
import sys
import os

from personal.personal import Person

async def main():
    girl = Person(name="李诗情", gender="Female", if_original=False)

    print(f"输入一段描述角色性格的话: ")
    try:
        description = input().strip()
    except EOFError:
        print("Error: No input provided.")
        return

    girl.init_big_five_profile(description)
    
    print(girl.personality)

if __name__ == "__main__":
    asyncio.run(main())