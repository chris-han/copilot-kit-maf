import { User as LucideUser, LucideProps } from 'lucide-react';

const UserLine = ({ className, ...props }: LucideProps) => {
  return <LucideUser className={className} {...props} />;
};

export default UserLine;