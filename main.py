import neat
import os


lost = 0
won = 0
smartest = 0
amountwon = []
def eval_genomes(genomes, config):
    players = []
    ge = []
    nets = []
    info = {}

    for genome_id, genome in genomes:
        players.append(genome_id)
        info[genome_id] = 0
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        genome.fitness = 0

    running = True
    while running:

        if len(players) == 0:
            break

        for i, genome in enumerate(players):
            output = nets[i].activate(inputs=[0.6])


            if output[0] >= 0.8 and output[0] <= 0.9:
                ge[i].fitness += 20
                print("won")
                global won
                if info[genome] == 0:
                    won += 1

                if info[genome] == 50:
                    global smartest
                    smartest += 1

                if info[genome] >= 50:
                    amountwon.append(info[genome])
                    players.remove(genome)

                info[genome] += 1

            else:
                global lost
                lost += 1
                ge[i].fitness -= 20
                amountwon.append(info[genome])
                players.remove(genome)


def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.run(eval_genomes, 10000)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
    average = 0
    for i in amountwon:
        average += i

    average /= len(amountwon)
    print(f"Won: {won}, Average: {average}")
    print(f"Super smort: {smartest}")
    print(f"Lost: {lost}")
