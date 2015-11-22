import subprocess
subprocess.call("git add .",shell=True)
ssh jingshi@me-dimitrovresearch.engr.utexas.edu
OWNkNmNmODBkZGU1ODE1MDRiZjdmMDNh

subprocess.call("git commit . -m \"automatic commit from python\" ",shell=True)
subprocess.call("git push heroku master",shell=True)
print "end"
