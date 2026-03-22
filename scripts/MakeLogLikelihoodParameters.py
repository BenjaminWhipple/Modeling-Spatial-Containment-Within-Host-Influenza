import pandas as pd
import numpy as np
from PyUtilities.GP_Likelihood import fit_gp_and_evaluate


### Gonzalez Virus

df = pd.read_csv("../Aggregate-Measurements/Viral_Titer_mock_pr8.csv")
X = df["Day"].to_numpy().reshape(-1,1)
#y = np.log10(((35*0.2*0.7*df["TCID50"]).to_numpy())**(2/3))
y = np.log10(((35*0.7*0.1*df["TCID50"]).to_numpy())**(2/3))
# We convert y to estimated particles.
# The 2/3 power approximately corrects for the difference in quantity expected in a 2d simulation and 3d reality.
# For simplicity, we ignore the incorporation of the uncertainty of the infectious particles.
# The measurements are originally in TCID50/g, so we also need to convert this to account for Balb/c mouse lungs being about 0.2 g https://www.nature.com/articles/s41598-023-44797-x
# Each PFU roughly corresponds to 20-50 viral particles. We take the midpoint of this range. (From Principles of Virology Volume 2)

#t_eval = np.linspace(0,10,101).reshape(-1,1)
#mu_plottable, cov_plottable = fit_gp_and_evaluate(X, y, t_eval)#, y_std=y_std)

#np.savez("SerializedObjects/GaussianProcessParameters/plottable_Virus_gp.npz", mu=mu_plottable, cov=cov_plottable, t_eval=t_eval)

t_eval = np.linspace(0,10,11).reshape(-1,1)
mu, cov = fit_gp_and_evaluate(X, y, t_eval)

np.savez("../data/gp_likelihoods/Virus_gp.npz", mu=mu, cov=cov, t_eval=t_eval)

### Toapanta Virus

df = pd.read_csv("../Aggregate-Measurements/ReorganizedData_ViralTiter.csv")
df = df[df["Age Group"] == "Adult"]
X = df["DPI"].to_numpy().reshape(-1,1)
#y = np.log10(((35*0.2*0.7*df["TCID50"]).to_numpy())**(2/3))

# Toapanta had about 4 ml of supernatant per viral measurement.
y = np.log10(((35*df["Viral Titer (Pfu/ml)"]*4+1e0).to_numpy())**(2/3))
# We convert y to estimated particles.
# The 2/3 power approximately corrects for the difference in quantity expected in a 2d simulation and 3d reality.
# For simplicity, we ignore the incorporation of the uncertainty of the infectious particles.
# The measurements are originally in TCID50/g, so we also need to convert this to account for Balb/c mouse lungs being about 0.2 g https://www.nature.com/articles/s41598-023-44797-x
# Each PFU roughly corresponds to 20-50 viral particles. We take the midpoint of this range. (From Principles of Virology Volume 2)

#t_eval = np.linspace(0,10,101).reshape(-1,1)
#mu_plottable, cov_plottable = fit_gp_and_evaluate(X, y, t_eval)#, y_std=y_std)

#np.savez("SerializedObjects/GaussianProcessParameters/plottable_Virus_gp.npz", mu=mu_plottable, cov=cov_plottable, t_eval=t_eval)

t_eval = np.linspace(0,10,11).reshape(-1,1)
mu, cov = fit_gp_and_evaluate(X, y, t_eval)

np.savez("../data/gp_likelihoods/Toapanta_Virus_gp.npz", mu=mu, cov=cov, t_eval=t_eval)

### CD8+ T Cells

df = pd.read_csv("../Aggregate-Measurements/CD8_TCells.csv")
X = df["Day"].to_numpy().reshape(-1,1)
y = (np.log10(((df["CD8"]).to_numpy()/2)**(2/3)))

# We divide by 2 to represent that we are only modeling half a lung.
# The 2/3 power approximately corrects for the difference in quantity expected in a 2d simulation and 3d reality.

t_eval = np.linspace(0,10,101).reshape(-1,1)
mu_plottable, cov_plottable = fit_gp_and_evaluate(X, y, t_eval)

np.savez("../data/gp_likelihoods/plottable_CD8_gp.npz", mu=mu_plottable, cov=cov_plottable, t_eval=t_eval)

t_eval = np.linspace(0,10,11).reshape(-1,1)
mu, cov = fit_gp_and_evaluate(X, y, t_eval)#, y_std=y_std)

np.savez("../data/gp_likelihoods/CD8_gp.npz", mu=mu, cov=cov, t_eval=t_eval)


### Toapanta IFN

# We approximate IFN I as IFN-beta.
# We approximate masses of IFN-beta as equivalent 20000 kDa
# We use the approximation that 1/2 moise lung is about 0.5 ml. -> Actually, we might need to use 4 ml.
# Finally, we use the approximation of (2/3) to convert to area from volume.

# so, C pg/ml ~ C * 0.5*(6.022 x 10^11 / 20000) for us.
# This approximates the number of particles.

df = pd.read_csv("../Aggregate-Measurements/IFN.csv")
X = df["DPI"].to_numpy().reshape(-1,1)
print(df)
#y = (np.log10((df["IFN-alpha (pg/ml)"]+df["IFN-beta (pg/ml) truncated"] + 1e-8) * 0.5 * (6.022 * (10**11))/20000)**(2/3))
y = (np.log10(((df["IFN-beta (pg/ml) truncated"]) * 4 * ((6.022 * (10**11))/20000))**(2/3) + 1e0))
print(y)

t_eval = np.linspace(0,10,101).reshape(-1,1)
mu_plottable, cov_plottable = fit_gp_and_evaluate(X, y, t_eval)#, y_std=y_std)

np.savez("../data/gp_likelihoods/plottable_IFN_gp.npz", mu=mu_plottable, cov=cov_plottable, t_eval=t_eval)

t_eval = np.linspace(0,10,11).reshape(-1,1)
mu, cov = fit_gp_and_evaluate(X, y, t_eval)

np.savez("../data/gp_likelihoods/IFN_gp.npz", mu=mu, cov=cov, t_eval=t_eval)