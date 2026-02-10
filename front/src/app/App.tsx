import { Header } from "@/layouts";
import { Outlet } from "@tanstack/react-router";

function App() {
  return (
    <main className="w-full bg-main-app/80 min-h-screen flex flex-col items-center max-w-480">
      <Header />
      <Outlet />
    </main>
  );
}

export default App;
