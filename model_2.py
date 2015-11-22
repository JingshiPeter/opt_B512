import pandas
import networkx
import pyomo.opt
import pyomo.environ as pe
import scipy
import itertools
import logging

#read in data
edge_df=pandas.read_csv('edge_data.csv')
node_df=pandas.read_csv('node_data.csv')
time_df=pandas.read_csv('time_window.csv')

# 15 customers + (512)
partial_nodes = ['321 W Ben White Blvd','3601 South Congress Avenue','3116 S Congress','1900 S 1st St',
                 '3508 S Lamar Blvd','4024 S Lamar Blvd','900 Austin Highlands Blvd','3003 S Lamar Blvd',
                 '4477 S Lamar Blvd','909 W Mary St','4960 US Route 290','1400 South Congress Avenue',
                 '1509 S Lamar Blvd','69 Rainey Street','1109 South Lamar Blvd','407 Radam Ln']
def assign_time_window():
    time_window_df=pandas.DataFrame()
    add=[]
    d=[]
    A_1=[]
    B_1=[]
    A_2=[]
    B_2=[]
    for node in node_df.nodes:
        if node in partial_nodes:
            add.append(node)
            d.append(time_df[time_df.Address == node].Demand.values[0])
            if pandas.isnull(time_df[time_df.Address == node].A_1.values):
                A_1.append(9)
            else:
                A_1.append(time_df[time_df.Address == node].A_1.values[0])
            if pandas.isnull(time_df[time_df.Address == node].B_1.values):
                B_1.append(18)
            else:
                B_1.append(time_df[time_df.Address == node].B_1.values[0])
            if pandas.isnull(time_df[time_df.Address == node].A_2.values):
                A_2.append(18)
                if pandas.isnull(time_df[time_df.Address == node].B_2.values):
                    B_2.append(9)
                else:
                    B_2.append(time_df[time_df.Address == node].B_2.values[0])
            else:
                A_2.append(time_df[time_df.Address == node].A_2.values[0])
                if pandas.isnull(time_df[time_df.Address == node].B_2.values):
                    B_2.append(18)
                else:
                    B_2.append(time_df[time_df.Address == node].B_2.values[0])
    time_window_df['Address']=add
    time_window_df['A_1']=A_1
    time_window_df['B_1']=B_1
    time_window_df['A_2']=A_2
    time_window_df['B_2']=B_2
    time_window_df['Demand']=d
    return time_window_df

timewindow_df=assign_time_window()   

def create_graph():
    """Turns the problem's input data into network data."""
    g=networkx.DiGraph()
    #create the node as the clients
    #create a starting and an end node
    for node in node_df.nodes:
        if node in partial_nodes:
            a1=timewindow_df[timewindow_df.Address == node].A_1.values[0]
            b1=timewindow_df[timewindow_df.Address == node].B_1.values[0]
            a2=timewindow_df[timewindow_df.Address == node].A_2.values[0]
            b2=timewindow_df[timewindow_df.Address == node].B_2.values[0]
            d = timewindow_df[timewindow_df.Address == node].Demand.values[0]
            g.add_node(node,A_1=a1,A_2=a2,B_1=b1,B_2=b2,Demand = d)
    #create the edge as the travel time from client i to client j
    #add an edge if time constraints allowed
    for i in edge_df.node1.unique():
        if i in partial_nodes:
            for j in edge_df[edge_df.node1 == i].node2:
                if j in partial_nodes:
                    cond1=timewindow_df[timewindow_df.Address == i].A_1.values[0] + edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600) <= timewindow_df[timewindow_df.Address == j].B_1.values[0]
                    cond2=timewindow_df[timewindow_df.Address == i].A_2.values[0] + edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600) <= timewindow_df[timewindow_df.Address == j].B_1.values[0]      
                    cond3=timewindow_df[timewindow_df.Address == i].A_1.values[0] + edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600) <= timewindow_df[timewindow_df.Address == j].B_2.values[0]
                    cond4=timewindow_df[timewindow_df.Address == i].A_2.values[0] + edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600) <= timewindow_df[timewindow_df.Address == j].B_2.values[0]  
                    if cond1:
                        g.add_edge(i, j, cost= edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600),VehicleCost=0,name="%s and %s"%(i,j)) 
                    elif cond2:
                        g.add_edge(i, j, cost= edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600),VehicleCost=0,name="%s and %s"%(i,j))
                    elif cond3:
                        g.add_edge(i, j, cost= edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600),VehicleCost=0,name="%s and %s"%(i,j))
                    elif cond4:
                        g.add_edge(i, j, cost= edge_df[(edge_df.node1 == i) & (edge_df.node2 == j)].travel_time_seconds.values[0]/float(3600),VehicleCost=0,name="%s and %s"%(i,j)) 
    #g.add_edge('407 Radam Ln','407 Radam Ln',cost =0, VehicleCost=0,name = "00")
    for node in node_df.nodes[1:]:
        if node in partial_nodes:
            g.edge['407 Radam Ln'][node]['VehicleCost']=3000
        
    return g
    
import itertools 
def powerset(iterable):
    """powerset([1,2,3]) --> ()(1,)(2,)(3,)(1,2)(1,3)(2,3)(1,2,3)"""
    s = list(iterable)
    return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s)+1))
            
def create_pyomo_network_lp_primal(g):
    model = pe.ConcreteModel()
    model.g = g

    # Create the problem data
    p = list(powerset(g.nodes()))
    model.power_set = pe.Set( initialize=p , dimen=None)
    
    ## Write some additional code here defining sets, parameters, variables
    #create index
    model.node_set=pe.Set(initialize = model.g.nodes())
    model.arc_set=pe.Set(initialize = model.g.edges())
    departure_arc=[]
    for arc in g.edges():
        if arc[0]=='407 Radam Ln':
            departure_arc.append(arc)
    model.departure_arc = pe.Set(initialize = departure_arc)
    return_arc=[]
    for arc in g.edges():
        if arc[1]=='407 Radam Ln':
            return_arc.append(arc)
    model.return_arc = pe.Set(initialize = return_arc)
    
    model.customer_arc= model.arc_set - model.departure_arc-model.return_arc
    model.depotnode = pe.Set(initialize = ['407 Radam Ln'])
    model.customer_node = model.node_set - model.depotnode
    model.vehicle = pe.Set(initialize =[0,1,2,3,4,5,6,7,8,9])
    costcf = [1,1.2,1.5,1.8,2,2,2,2,2,2]
    vcostcf = [1,1.5,2,3,3,3,3,3,3,3]
    car_capacity = [100,200,300,500,600,600,600,600,600,600]
    def indexrule(model):
        return [(v,i,j) for v in model.vehicle for i,j in model.arc_set]
    model.veh_arc_set=pe.Set(dimen=3, initialize = indexrule)
    #create parameters                                             
    model.cost = pe.Param(model.veh_arc_set, initialize = lambda model,v,i,j: costcf[v]* model.g.edge[i][j]['cost'], mutable = True)
    model.vcost = pe.Param(model.veh_arc_set, initialize = lambda model,v,i,j: vcostcf[v]*model.g.edge[i][j]['VehicleCost'], mutable = True)
    model.demand = pe.Param(model.customer_node, initialize = lambda model,n: model.g.node[n]['Demand'], mutable = True)
    model.loadingtime = pe.Param(model.customer_node, initialize = 0.25, mutable = True)
    model.Tij = pe.Param(model.arc_set, initialize = lambda model,i,j: model.g.edge[i][j]['cost'], mutable = True)
    model.big_M = 10000
    model.K=len(model.vehicle)
    #create variables
    model.x=pe.Var(model.veh_arc_set, domain = pe.Binary)
    model.y=pe.Var(model.arc_set, domain = pe.Binary)
    model.iv = pe.Var(model.customer_node, domain = pe.Binary)
    model.t=pe.Var(model.customer_node, domain = pe.NonNegativeReals)
    model.t_depot = 9
    model.wt = pe.Var(model.arc_set, domain = pe.NonNegativeReals)
    ## Define objective and constraints
    def pset_rule(model, *q):
        if len(q) == 0 or ('407 Radam Ln' in q):
            return pe.Constraint.Skip
        lhs = 0
        empty = True
        for i in q:
            for j in q:
                if model.g.has_edge(i,j):
                    empty = False
                    lhs = lhs + model.y[(i,j)]
        if empty:
            return pe.Constraint.Skip
        return lhs <= len(q) - 1
    model.PSetConst = pe.Constraint(model.power_set, rule=pset_rule)
    
    ## Write some code here defining the objective function, rest of constraints etc 
    #define objective function
    #model.OBJ = pe.Objective(expr = pe.summation(model.cost, model.x), sense=pe.minimize)
    model.OBJ = pe.Objective(expr = pe.summation(model.cost, model.x)+pe.summation(model.vcost,model.x)+pe.summation(model.wt), sense=pe.minimize)
    #define flow constraint
    def flow_bal(model,v,n):
        return (pe.summation(model.x, index = [(a,i,j) for a,i,j in model.veh_arc_set if j==n if a == v]) - pe.summation(model.x, index = [(a,i,j) for a,i,j in model.veh_arc_set if i==n if a == v]) == 0)
    model.FlowConst_v1=pe.Constraint(model.vehicle, model.node_set, rule=flow_bal)
    #define the second constraint
    def sec_const(model,i,j):
        return (pe.summation(model.x, index = [(v,i,j) for v in model.vehicle]) == model.y[(i,j)])
    model.ArcConst = pe.Constraint(model.arc_set, rule = sec_const)
    def third_const(model,j):
        return (pe.summation(model.y, index=[arc for arc in model.arc_set if arc[1]==j])==1)
    model.ThirdConst = pe.Constraint(model.customer_node, rule = third_const)
    def forth_const(model,j):
        return (pe.summation(model.y, index =[arc for arc in model.arc_set if arc[0]==j])==1)
    model.ForthConst = pe.Constraint(model.customer_node, rule = forth_const)
    def fif_const(model,depot):
        return (pe.summation(model.y,index = [arc for arc in model.arc_set if arc[1]==depot])<=model.K)
    model.FifConst = pe.Constraint(model.depotnode, rule = fif_const)
    def six_const(model,depot):
        return (pe.summation(model.y,index = [arc for arc in model.arc_set if arc[0]==depot])<=model.K)
    model.SixConst = pe.Constraint(model.depotnode, rule = six_const)
    def y_flow(model,depot):
        return (pe.summation(model.y,index = [arc for arc in model.arc_set if arc[1]==depot]) == pe.summation(model.y,index = [arc for arc in model.arc_set if arc[0]==depot]))
    model.yflowconst = pe.Constraint(model.depotnode, rule = y_flow)
    def const7(model,k):
            summ=0
            for j in model.customer_node:
                arclist = [arc for arc in model.arc_set if arc[1]==j]
                for arc in arclist:
                    summ=summ+model.demand[j]*model.x[(k,arc[0],arc[1])]
            return (summ <= car_capacity[k])
    model.const7 = pe.Constraint(model.vehicle, rule = const7)    
    ######################################################################
    #define timewindow constraint
    def timewindow_const1(model,n):
        return ( model.g.node[n]['A_1']*model.iv[n] +model.g.node[n]['A_2']*(1-model.iv[n]) <= model.t[n] )
    model.timeconst1 = pe.Constraint(model.customer_node, rule = timewindow_const1)
    def timewindow_const2(model,n):
        return (  model.t[n] <= model.g.node[n]['B_1']*model.iv[n]+model.g.node[n]['B_2']*(1-model.iv[n]) )
    model.timeconst2 = pe.Constraint(model.customer_node, rule = timewindow_const2)
    def finconst(model,i,j):    
        return ( model.t[i]+model.loadingtime[i]+model.Tij[(i,j)]+model.wt[(i,j)]-model.t[j] <= model.big_M*(1-model.y[(i,j)]) )
    model.finconst = pe.Constraint(model.customer_arc,rule = finconst)    
    def finconst2(model,i,j):    
        return ( model.t[i]+model.loadingtime[i]+model.Tij[(i,j)]+model.wt[(i,j)]-model.t[j] >= -model.big_M*(1-model.y[(i,j)]) )
    model.finconst2 = pe.Constraint(model.customer_arc,rule = finconst2)
    def finconst3(model,n):
        return (model.t_depot + model.Tij[('407 Radam Ln',n)]+model.wt[('407 Radam Ln',n)] <= model.t[n])
    model.finconst3 = pe.Constraint(model.customer_node,rule = finconst3)
    
    
    # Solve the model
    model.create()
    model= resolve_mip(model)

    # Print the model objective
    print 'Primal objective function value:', model.OBJ()
    return model

def resolve_mip(model, tee=False):
    model.preprocess()
    solver = pyomo.opt.SolverFactory('cplex')
    results = solver.solve(model, tee=tee, keepfiles=False, options="mip_tolerances_integrality=1e-9 mip_tolerances_mipgap=0")
    
    if (results.solver.status != pyomo.opt.SolverStatus.ok):
        logging.warning('Check solver not ok?')
    if (results.solver.termination_condition != pyomo.opt.TerminationCondition.optimal):  
        logging.warning('Check solver optimality?')

    model.load(results)
    
    model.g.OBJ_val = model.OBJ()
     
    return model


def print_solution(model):
    ## Write some code here
    for v in model.vehicle:
        for i,j in model.arc_set:
            if model.x[(v,i,j)] ==1:
                print 'Vehicle %d travels on edge (%s,%s)'%(v,i,j)
    print '----------'
    for n in model.customer_node:
        print 'Arrive at ',n," at ", model.t[n].value
        print 'Loading time for %s is '%(n), model.loadingtime[n].value
    print '----------'
    print 'Total cost = ', model.OBJ()



g = create_graph()

mPrimal = create_pyomo_network_lp_primal(g)

print_solution(mPrimal)
