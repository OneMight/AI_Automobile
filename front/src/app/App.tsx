import { Header } from "@/layouts";
import { Outlet } from "@tanstack/react-router";

function App() {
  return (
    <main className="w-full bg-main-app/80 min-h-screen flex flex-col items-center max-w-480 gap-15 relative">
      <Header />
      <div className="w-full flex justify-center items-center">
        <Outlet />
      </div>
    </main>
  );
}

export default App;
