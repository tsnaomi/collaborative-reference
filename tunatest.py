from tunaagents import *

litspk = LitSpeaker(allgames, produce_all=True)

acctlv0_A = []
acctlv2_A = []
acctlvx_A = []
acclv0_A = []
acclv2_A = []
acclvx_A = []

acctlv0_N = []
acctlv2_N = []
acctlvx_N = []
acclv0_N = []
acclv2_N = []
acclvx_N = []


for i in range(7):
    print "training {0} game(s) with literal semantics".format(i+1)
    annlit = ANNListener(allgames[:(i+1)], hidden_dim=50,
                         check_literal="all")
    annlit.learn(litspk)
    acctlv0_A.append(EvaluateListener(annlit, tlevel0))
    acctlv2_A.append(EvaluateListener(annlit, tlevel2))
    acctlvx_A.append(EvaluateListener(annlit, tlevelx))
    acclv0_A.append(EvaluateListener(annlit, level0))
    acclv2_A.append(EvaluateListener(annlit, level2))
    acclvx_A.append(EvaluateListener(annlit, levelx))

    print "training {0} game(s) without literal semantics".format(i+1)
    annlit = ANNListener(allgames[:(i+1)], hidden_dim=50,
                         check_literal=None)
    annlit.learn(litspk)
    acctlv0_N.append(EvaluateListener(annlit, tlevel0))
    acctlv2_N.append(EvaluateListener(annlit, tlevel2))
    acctlvx_N.append(EvaluateListener(annlit, tlevelx))
    acclv0_N.append(EvaluateListener(annlit, level0))
    acclv2_N.append(EvaluateListener(annlit, level2))
    acclvx_N.append(EvaluateListener(annlit, levelx))



print "With literal semantics (0-2-x true-all)"
print acctlv0_A
print acctlv2_A
print acctlvx_A
print acclv0_A
print acclv2_A
print acclvx_A

print "Without literal semantics (0-2-x true-all)"
print acctlv0_N
print acctlv2_N
print acctlvx_N
print acclv0_N
print acclv2_N
print acclvx_N
