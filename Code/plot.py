import numpy as np
import matplotlib.pyplot as plt
# Convergence test plot
a = np.arange(5.0,71.0,5)
b = np.array([9.042540594444626878e-01,
2.194904204578775053e-01,
2.074143348600335224e-01,
7.434653270072190401e-02,
8.447904285618772213e-02,
4.036253568006076264e-02,
4.701801862372274182e-02,
2.620697023860175945e-02,
3.060449968470332210e-02,
1.875112255341825768e-02,
2.184258555898753451e-02,
1.426449505896217398e-02,
1.655606546941080018e-02,
1.131967852152349768e-02])
y = a ** -2 *43
fig1, ax = plt.subplots()
plt.loglog(a,b,label=r'$e_u(h,\tau)$')
plt.loglog(a,y, label=r'$O(\frac{1}{N_t^2})$')
ax.set_xlabel('Mesh Size O(h) and O('+r'$\tau$'+')')
ax.set_ylabel("Error Norms")
plt.legend()
plt.title('Log-log Plot of Mesh Size over Errors')
plt.savefig('convergence.png')

# Solution Plots
N_t = 60
x = np.linspace(0,2,N_t)
x_sol = np.loadtxt("x_sol_15")
x_ana = np.loadtxt("x_ana_15")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol[1:], label='Numerical Solution')
fig1.plot(x, x_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution u')
plt.ylim(-1.25,1.25)
plt.legend()
plt.title('x=0.5')
plt.savefig('x15.png')
plt.show()
xp_sol = np.loadtxt("xp_sol_15")
xp_ana = np.loadtxt("xp_ana_15")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, xp_sol[1:], label='Numerical Solution')
fig2.plot(x, xp_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('x=0.5')
plt.savefig('xp15.png')
plt.show()
N_t = 60
x = np.linspace(0,2,N_t)
x_sol = np.loadtxt("x_sol_10")
x_ana = np.loadtxt("x_ana_10")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol[1:], label='Numerical Solution')
fig1.plot(x, x_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution u')
plt.ylim(-1.25,1.25)
plt.legend()
plt.title('x=1/3')

plt.savefig('x10.png')
plt.show()
xp_sol = np.loadtxt("xp_sol_10")
xp_ana = np.loadtxt("xp_ana_10")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, xp_sol[1:], label='Numerical Solution')
fig2.plot(x, xp_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('x=1/3')

plt.savefig('xp10.png')
plt.show()
N_t = 60
x = np.linspace(0,2,N_t)
x_sol = np.loadtxt("x_sol_5")
x_ana = np.loadtxt("x_ana_5")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol[1:], label='Numerical Solution')
fig1.plot(x, x_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution u')
plt.ylim(-1.25,1.25)
plt.legend()
plt.title('x=1/6')

plt.savefig('x5.png')
plt.show()
xp_sol = np.loadtxt("xp_sol_5")
xp_ana = np.loadtxt("xp_ana_5")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, xp_sol[1:], label='Numerical Solution')
fig2.plot(x, xp_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('x=1/6')

plt.savefig('xp5.png')
plt.show()
N_t = 60
x = np.linspace(0,2,N_t)
x_sol = np.loadtxt("x_sol_20")
x_ana = np.loadtxt("x_ana_20")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol[1:], label='Numerical Solution')
fig1.plot(x, x_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution u')
plt.ylim(-1.25,1.25)
plt.legend()
plt.title('x=2/3')

plt.savefig('x20.png')
plt.show()
xp_sol = np.loadtxt("xp_sol_20")
xp_ana = np.loadtxt("xp_ana_20")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, xp_sol[1:], label='Numerical Solution')
fig2.plot(x, xp_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('x=2/3')

plt.savefig('xp20.png')
plt.show()
N_t = 60
x = np.linspace(0,2,N_t)
x_sol = np.loadtxt("x_sol_25")
x_ana = np.loadtxt("x_ana_25")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol[1:], label='Numerical Solution')
fig1.plot(x, x_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution u')
plt.ylim(-1.25,1.25)
plt.legend()
plt.title('x=5/6')

plt.savefig('x25.png')
plt.show()
xp_sol = np.loadtxt("xp_sol_25")
xp_ana = np.loadtxt("xp_ana_25")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, xp_sol[1:], label='Numerical Solution')
fig2.plot(x, xp_ana[1:], label='Analytic Solution')
plt.xlabel('Time t')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('x=5/6')

plt.savefig('xp25.png')
plt.show()
N_x = 31
x = np.linspace(0,1,N_x)
x_sol = np.loadtxt("t_sol_10")
x_ana = np.loadtxt("t_ana_10")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol, label='Numerical Solution')
fig1.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution u')
plt.ylim(0,1)
plt.legend()
plt.title('t=1/3 s')

plt.savefig('t10.png')
plt.show()
x_sol = np.loadtxt("tp_sol_10")
x_ana = np.loadtxt("tp_ana_10")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, x_sol, label='Numerical Solution')
fig2.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('t=1/3 s')

plt.savefig('tp10.png')
plt.show()
N_x = 31
x = np.linspace(0,1,N_x)
x_sol = np.loadtxt("t_sol_20")
x_ana = np.loadtxt("t_ana_20")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol, label='Numerical Solution')
fig1.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution u')
plt.ylim(-1,0)
plt.legend()
plt.title('t=2/3 s')

plt.savefig('t20.png')
plt.show()
x_sol = np.loadtxt("tp_sol_20")
x_ana = np.loadtxt("tp_ana_20")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, x_sol, label='Numerical Solution')
fig2.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('t=2/3 s')

plt.savefig('tp20.png')
plt.show()
N_x = 31
x = np.linspace(0,1,N_x)
x_sol = np.loadtxt("t_sol_30")
x_ana = np.loadtxt("t_ana_30")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol, label='Numerical Solution')
fig1.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution u')
plt.ylim(-1,0)
plt.legend()
plt.title('t=1 s')

plt.savefig('t30.png')
plt.show()
x_sol = np.loadtxt("tp_sol_30")
x_ana = np.loadtxt("tp_ana_30")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, x_sol, label='Numerical Solution')
fig2.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('t=1 s')

plt.savefig('tp30.png')
plt.show()
N_x = 31
x = np.linspace(0,1,N_x)
x_sol = np.loadtxt("t_sol_40")
x_ana = np.loadtxt("t_ana_40")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol, label='Numerical Solution')
fig1.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution u')
plt.ylim(-1,0)
plt.legend()
plt.title('t=4/3 s')
plt.savefig('t40.png')
plt.show()
x_sol = np.loadtxt("tp_sol_40")
x_ana = np.loadtxt("tp_ana_40")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, x_sol, label='Numerical Solution')
fig2.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('t=4/3 s')
plt.savefig('tp40.png')
plt.show()
N_x = 31
x = np.linspace(0,1,N_x)
x_sol = np.loadtxt("t_sol_50")
x_ana = np.loadtxt("t_ana_50")
fig1 = plt.subplot(1,1,1)
fig1.plot(x, x_sol, label='Numerical Solution')
fig1.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution u')
plt.ylim(0,1)
plt.legend()
plt.title('t=5/3 s')
plt.savefig('t50.png')
plt.show()
x_sol = np.loadtxt("tp_sol_50")
x_ana = np.loadtxt("tp_ana_50")
fig2 = plt.subplot(1,1,1)
fig2.plot(x, x_sol, label='Numerical Solution')
fig2.plot(x, x_ana, label='Analytic Solution')
plt.xlabel('Position x')
plt.ylabel('Solution p')
plt.ylim(0,45)
plt.legend()
plt.title('t=5/3 s')
plt.savefig('tp50.png')
plt.show()
