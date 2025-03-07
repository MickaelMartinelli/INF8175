import subprocess

n = 10 
wins = {'1' : 0, '2' : 0}

for i in range(n):
    print("Partie", i+1)
    process = subprocess.Popen(["python", "main_divercite.py", "-t", "local", "my_player_test.py", "basic_minimax.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stderr, stdout = process.communicate()

    '''lines = stdout.strip().split("\n")

    if stderr:

        print(f"Erreurs :\n{stderr}")

    else:

        score1 = lines[-6][-2:]
        print("Score du joueur 1 :", score1)

        score2 = lines[-5][-2:]
        print("Score du joueur 2 :", score2)

        winner = lines[-4][-1]
        print("Joueur gagnant : joueur", winner)

        wins[winner]+=1
        print("-" * 50)

print(wins)'''