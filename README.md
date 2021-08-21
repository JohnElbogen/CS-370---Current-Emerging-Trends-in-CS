# CS-370
Briefly explain the work that you did on this project: What code were you given? What code did you create yourself?
In this project, I was given code for a class representing experience and the treasure maze, that contained methods for the actual game state and experience replay object. Also, part of the q-training model was provided and examples of how the code runs and can be tested, plus testing methods.
The code I wrote myself was the following:
```
    for epoch in range(n_epoch):

        loss = 0.0
        
        Agent_cell = random.choice(qmaze.free_cells)
        
        qmaze.reset(Agent_cell)
        game_over = False
        
        envstate = qmaze.observe()

        n_episodes = 0
        
        while not game_over:
            #print("start loop")
            valid_actions = qmaze.valid_actions()
            if not valid_actions: break
            previous_envstate = envstate         
            if (np.random.rand() < epsilon):
                action = random.choice(valid_actions)
            else:
                action = np.argmax(experience.predict(previous_envstate))
            envstate, reward, game_status = qmaze.act(action)
            if game_status == 'win':
                win_history.append(1)
                game_over = True
            elif game_status == 'lose':
                win_history.append(0)
                game_over = True
            else:
                game_over = False
                
            episode = [previous_envstate, action, reward, envstate, game_over]
            experience.remember(episode)
            n_episodes+=1
            inputs, targets = experience.get_data(data_size=50)
            h = model.fit(inputs, targets, epochs=4, batch_size=8, verbose=0)
            loss = model.evaluate(inputs, targets, verbose=0)

        if len(win_history) > hsize:
            win_rate = sum(win_history[-hsize:]) / hsize    
```
Connect your learning from throughout this course to the larger field of computer science:
What do computer scientists do and why does it matter?
Computer scientists cover a broad range of computer applications, from research, software engineering, and discovering/exploring technological solutions.

How do I approach a problem as a computer scientist?
As a computer scientist, I look at problems in a way where I grasp the elements as much as I can. I tend to take the simplest approach, with the most effective outcome. Also, I try to understand the problem as thoroughly as possible. Then I utilize every resource I can to find solutions to that problem.

What are my ethical responsibilities to the end-user and the organization?
My ethical responsibilities are to be aware and knowledgeable to produce non-malicious/non-harmful solutions to the end user. To the organization, my responsibilities are to apply my knowledge and meet expectations while following my moral boundary.
