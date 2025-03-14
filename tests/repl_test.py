from slimagents import Agent, run_demo_loop

default_memory = [
        {
            "role": "user",
            "content": "Is the sky blue?"
        },
        {
            "role": "assistant",
            "content": "YES"
        },
        {
            "role": "user",
            "content": "Is the sky blue?"
        },
        {
            "role": "assistant",
            "content": "NO"
        },
        {
            "role": "user",
            "content": "Is the sky blue?"
        },
        {
            "role": "assistant",
            "content": "YES"
        },
]

agent = Agent(
    instructions="You follow communication patterns and like to play games with the user.",
    temperature=0.0,
    memory=default_memory,
)


run_demo_loop(agent)
