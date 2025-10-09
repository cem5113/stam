# services/auth.py
from __future__ import annotations
import os
from typing import Literal

Role = Literal["Amir", "Kullanıcı"]

_ROLES = ("Amir", "Kullanıcı")

def get_role() -> Role:
    """Öncelik: Streamlit session_state → ENV(APP_ROLE) → 'Kullanıcı'."""
    try:
        import streamlit as st  # yalnızca varsa
        r = st.session_state.get("role")
        if r in _ROLES:
            return r  # type: ignore
    except Exception:
        pass
    r_env = os.environ.get("APP_ROLE", "Kullanıcı")
    return r_env if r_env in _ROLES else "Kullanıcı"  # type: ignore

def set_role(role: Role) -> None:
    try:
        import streamlit as st
        st.session_state["role"] = role
    except Exception:
        os.environ["APP_ROLE"] = role  # CLI/başka ortam için

def can_approve() -> bool:
    return get_role() == "Amir"

def role_selector_in_sidebar(default: Role = "Kullanıcı") -> Role:
    """Streamlit varsa sidebar’da rol seçici render eder; seçimi döndürür."""
    try:
        import streamlit as st
        cur = get_role() or default
        role = st.sidebar.selectbox("Rol", _ROLES, index=_ROLES.index(cur))
        set_role(role)  # persist
        st.sidebar.caption("Rol: Amir → onay/yetki; Kullanıcı → yalnız görüntüleme.")
        return role  # type: ignore
    except Exception:
        return get_role()
