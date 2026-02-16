import { Header, LoginedHeader } from "@/layouts";
import { useUser } from "@/lib/useUser";
import { Outlet } from "@tanstack/react-router";

function App() {
  const { user, isLoading } = useUser();
  return (
    <main className="w-full bg-main-app/80 min-h-screen flex flex-col items-center max-w-480 gap-15 relative">
      {!isLoading && user ? <LoginedHeader /> : <Header />}
      <div className="w-full flex justify-center items-center">
        <Outlet />
      </div>
    </main>
  );
}

export default App;
