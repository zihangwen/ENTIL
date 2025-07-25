import numpy as np

class Clone(object): # representing a group of nodes with shared identity and fitness.
    def __init__(self, cid, N, fitness = 1):
        self.cid = cid
        self.N = N
        self.count = 0
        self.fitness = fitness
    
        self.nodes = [] # stores node ID's
        self.index = {} # dict, maps nodes to indexes
    
    def add(self, node):
        assert node not in self.index
        self.nodes += [node]
        self.index[node] = self.count
        self.count += 1
        
    def remove(self, node):
        assert node in self.index
        self.count -= 1
        idx = self.index[node] #find index
        
        self.nodes[idx] = self.nodes[self.count] # swap with last 
        self.index[self.nodes[self.count]] = idx # replace index
        
        self.nodes.pop()
        self.index.pop(node)
    
    def sample(self):
        return self.nodes[np.random.randint(self.count)] # Returns a random node from the clone using uniform sampling.
        
    def total_fitness(self):
        return self.count * self.fitness
        
class Population(object):
    def __init__(self, G):
        self.G = G # dict with node mapping to neighbours
        self.N = len(G)
        self.clones = {0 : Clone(0, self.N)} # 0 is the starting clone

        for i in range(self.N):
            self.clones[0].add(i) # add all nodes to clone 0
            
        self.node_to_cid = np.zeros(self.N).astype(int) # map all to clone 0
        self.mut_cnt = 1 # assigning new clone IDs during mutation events.
        
    def bd(self):
        cids, fits = [], []
        for cid in self.clones:
            cids += [cid]
            fits += [self.clones[cid].total_fitness()]
        
        cid_birth = np.random.choice(cids, p = np.array(fits) / np.sum(fits)) # choose clone to duplicate
        n_birth = self.clones[cid_birth].sample() # select random node in the birht clone 
        
        n_death = np.random.choice(self.G[n_birth]) # select neighrbour to die
        cid_death = self.node_to_cid[n_death] # find cloneid of the death neighbour
        self.clones[cid_death].remove(n_death) 
        
        if self.clones[cid_death].count == 0:
            self.clones.pop(cid_death) # pop the clone if no nodes
        
        self.clones[cid_birth].add(n_death) # copy birth node identity into dead node
        self.node_to_cid[n_death] = self.node_to_cid[n_birth]  # copy ID as well
        
        return n_birth, n_death
    
    def mutate(self, node):
        cid = self.mut_cnt # assigns new cloneid
        
        clone = Clone(cid, self.N, fitness = 1) # create new clone with id and add node to it
        clone.add(node)
        self.clones[cid] = clone # add to list of clones
        self.clones[self.node_to_cid[node]].remove(node) # remove node from previous clone
        
        self.node_to_cid[node] = cid # update mapping
        self.mut_cnt += 1
        
        return cid

###########################################################################
def load_G(f):
    G = dict()
    el = np.loadtxt(f).astype(int)
    for edge in el:
        n1, n2 = edge
        if n1 in G:
            G[n1] += [n2]
        else:
            G[n1] = [n2]

        if n2 in G:
            G[n2] += [n1]
        else:
            G[n2] = [n1]
    return G

###########################################################################
def fitness_rank(score, s = 1):
    p = score.argsort().argsort()
    return (1 - s + (2 * s * (p) ) / (len(p) - 1)).clip(min = 0)

# p --> who's fitter, s --> how much that fitness matters (0 vs 1)

###########################################################################
class Pop_wrapper(object):
    def __init__(self, pop, tot_num_sol, tot_num_hyp, cur_num_sol):
        self.tot_num_sol = tot_num_sol
        self.tot_num_hyp = tot_num_hyp
        self.cur_num_sol = cur_num_sol
        
        self.pop = pop
        for node in range(1, self.pop.N):
            self.pop.mutate(node) # mutate every node but 0, so u start with one node per clone
        
        while len(self.pop.clones) > cur_num_sol: 
            n_birth, n_death = self.pop.bd() # run bd until population is controlled

        self.clone_to_sol = {}
        self.clone_to_hyp = {}
        
        self.sol_pool = np.arange(self.tot_num_sol) # soln indices
        self.open = np.array(self.tot_num_sol * [True])
        temp = np.random.choice(self.sol_pool, self.cur_num_sol, replace = False) # Randomly select cur_num_sol unique solution slots from the pool to assign to active clones.

        for idx, cid in enumerate(self.pop.clones):
            self.clone_to_sol[cid] = temp[idx]
            if self.tot_num_hyp > 0:
                self.clone_to_hyp[cid] = np.random.randint(self.tot_num_hyp)
            self.open[temp[idx]] = False
        
        self.visit_cnts = np.zeros((tot_num_hyp, tot_num_sol)) # how many used
        self.visit_fits = np.nan * np.zeros((tot_num_hyp, tot_num_sol)) # fitness updates
        
    def update(self, rewards, t_gen = 1):
        ### assign fitness
        fitness = fitness_rank(rewards, 0.2)
        for idx, cid in enumerate(self.pop.clones):
            self.pop.clones[cid].fitness = fitness[idx].item()
            self.visit_cnts[self.clone_to_sol[cid]] += 1
            self.visit_fits[self.clone_to_sol[cid]] = rewards[idx].item()
        
        ### birth death
        for t in range(int(t_gen * self.pop.N) ):
            n_birth, n_death = self.pop.bd()
            if len(self.pop.clones) < self.cur_num_sol: # if clone count drops below target mutate to replenish diversity
                self.pop.mutate(n_birth)
    
        ### add mutations 
        new_mut_cnt = 0 
        for cid in list(self.clone_to_sol.keys()): # go thru every cid
            if cid not in self.pop.clones:
                self.open[self.clone_to_sol[cid]] = True # if dead, mark it open again
                self.clone_to_sol.pop(cid) # delete from relevant maps
                if self.clone_to_hyp:
                    self.clone_to_hyp.pop(cid)
                new_mut_cnt += 1 # how many new clones need to be assigned
        
        ### assign to sol       
        temp = np.random.choice(self.sol_pool[self.open], new_mut_cnt, replace = False)
        '''pick new_mut_cnt random, unused solution slots from sol_pool to assign to new clones'''
        idx = 0
        for cid in self.pop.clones:
            if cid not in self.clone_to_sol:
                self.clone_to_sol[cid] = temp[idx]
                if self.tot_num_hyp > 0:
                    self.clone_to_hyp[cid] = np.random.randint(self.tot_num_hyp)
                self.open[temp[idx]] = False
                idx += 1
        
    @property
    def sol_current(self):
        return [(self.clone_to_sol[cid]) for cid in self.clone_to_sol]

    @property
    def freq_current(self):
        return [self.pop.clones[cid].count / self.pop.N for cid in self.clone_to_sol]
