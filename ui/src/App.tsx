import { NavLink, Route, Routes } from "react-router-dom";
import { HealthBanner } from "./components";
import { usePoll } from "./hooks";
import { api, type Health } from "./api";
import { useCallback, useEffect, useState } from "react";
import Assistants from "./pages/Assistants";
import Talk from "./pages/Talk";
import Calls from "./pages/Calls";
import Tools from "./pages/Tools";
import Squads from "./pages/Squads";
import Evals from "./pages/Evals";
import System from "./pages/System";

const NAV = [
  { to: "/", label: "Assistants", end: true },
  { to: "/talk", label: "Talk" },
  { to: "/calls", label: "Calls" },
  { to: "/tools", label: "Tools" },
  { to: "/squads", label: "Squads" },
  { to: "/evals", label: "Evals" },
  { to: "/system", label: "System" },
];

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);

  // Polled rather than fetched once. A service that goes down while the UI is
  // open is exactly the situation where the banner matters, and a static
  // banner is worse than none because it looks authoritative.
  const check = useCallback(async () => {
    try {
      setHealth(await api.health());
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    void check();
  }, [check]);

  usePoll(check, 10_000);

  return (
    <div className="shell">
      <nav className="sidebar">
        <div className="brand">
          <img src="/logo.svg" alt="" />
          <div>
            Voice Agent
            <small>local control plane</small>
          </div>
        </div>

        {NAV.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.end}
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            {item.label}
          </NavLink>
        ))}
      </nav>

      <main className="main">
        <HealthBanner health={health} />
        <Routes>
          <Route path="/" element={<Assistants />} />
          <Route path="/talk" element={<Talk />} />
          <Route path="/calls" element={<Calls />} />
          <Route path="/tools" element={<Tools />} />
          <Route path="/squads" element={<Squads />} />
          <Route path="/evals" element={<Evals />} />
          <Route path="/system" element={<System />} />
        </Routes>
      </main>
    </div>
  );
}
