---
tags: []
---

# Physics components interface: return values

- **Status**: valid ~~| superseded | deprecated~~
- **Authors**: Magdalena Luz (@halungge)
- **Created**: 2024-07-23
- **Updated**: YYYY-MM-DD

In the context of [use case/user story u], facing [concern c] we decided for [option o] and 
neglected [other options], to achieve [system qualities/desired consequences], 
accepting [downside/undesired consequences].

## Context
We aim at making `icon4py` a flexible model code which endorses modularity and extensibility, and makes it
easy for future users and developers to add new components, replace existing ones. This includes the 
possibility of experimenting with the coupling of different physics components, revising 
the order in which they are run.

In the original [ICON code](https://gitlab.dkrz.de/icon/icon-model) physics components may do both: compute tendencies and as well as
update the model state (update diagnostic and prognostic variables directly). We believe that this behavior
is very intransparent, makes the code inflexibel and the individual components and their order tightly coupled.

We want to provide a cleaner and more transparent structure.

**Concerns**

- If a component depends on the tendencies produced by another component, it is unclear whether the effect of this other component can be cleanly isolated.

]

## Decision

All **physics components return tendencies**. It is **forbidden to change input diagnostic and prognostic 
fields** ever inside a component. The model state is updated in a separate step after all tendencies have been computed.
In that step, the developer can decide which tendencies (produced by which component) need to be applied to the state.

Let's consider three physics components `f`, `g`, `h` which compute tendencies for the fields `rho`, `theta`, `T`.

```python
state = (rho, theta, w, T) # contains diagnostic and prognostic variables at time t

f(rho, T, w) -> ddt_rho, ddt_T # rho, T, w are input only
g(rho, T, theta) -> ddt_theta, ddt_T # rho, theta, are input only
h(rho, T, ddt_T) -> ddt_T # rho, theta, ddt_theta are input only

```
All three components produce tendencies for `T`, these need to be kept separate until the state is finally updated.
A components might depend one each other which can be made transparent through a dependency on the tendencies: in the example 
h might incorporate changes from `f` and `g` through the dependencies on `ddt_T`. This dependency has to the component as
explicit argument. The component `h` can then decide how to incorporate the tendencies from `f` and `g` or both.
Hence `ddt_T` needs to be some struct that lets the user access the tendencies computed by `f` and `g` individually.


## Consequences
Several physics components might produce tendencies for the same field: for example microphysics and radiation
produce tendencies for temperature. These individual tendencies need to be kept apart until the state is finally updated
prior to the next dynamics step. This leads to a more complex structure of the model state, especially the tendency fields.

On the positive side:
It should be easier to understand what exactly is in the model state and what tendencies issued from another
physics component are used where.

On the negative side: keeping tendencies produced by the individual components separate is more memory intensive
and requires an explicit update step as well as a more complex structure to track and handle the tendency fields.
.) and negative (e.g., compromising quality attribute, follow-up decisions required, ...)] outcomes of this decision. ]

## Alternatives Considered

### update state (prognostics, diagnostics) directly


```python
state = (rho, theta, w, T) # contains diagnostic and prognostic variables at time t

f(rho, T, w) -> rho, T # rho, T, w are input only
g(rho, T, theta) -> theta, T # rho, theta, are input only
h(rho, T) -> T # rho, theta, ddt_theta are input only
```
- Good, because no extra handling and bookkeeping of tendencies is needed.
- Bad, because it is unclear outside of a component whether state has changed, and what changes it incorporates.
- Bad, because the input value for a component changes on depending on the order of the components.

### mixture of both

```python
state = (rho, theta, w, T) # contains diagnostic and prognostic variables at time t

f(rho, T, w) -> ddt_rho, ddt_T # rho, T, w are input only
g(rho, T, theta) -> ddt_theta, T # rho, theta, are input only
h(rho, T) -> T # rho, theta, ddt_theta are input only
```

- Bad, unclear interface.

## References <!-- optional -->