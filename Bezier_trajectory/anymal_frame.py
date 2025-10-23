import example_robot_data
anymal = example_robot_data.load("anymal")
for i, f in enumerate(anymal.model.frames):
    print(i, f.name)
